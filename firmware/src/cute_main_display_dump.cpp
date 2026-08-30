#include <Arduino.h>
#include <math.h>
#include <SPI.h>
#include <Adafruit_GFX.h>
#include <Adafruit_ST7735.h>
#include "esp_camera.h"
#include "camera_pins.h"
#include "noodle_serial.h"

#include "freertos/FreeRTOS.h"
#include "freertos/task.h"
#include "freertos/semphr.h"

#ifndef NOODLE_USE_NONE
#define NOODLE_USE_NONE
#endif

#ifndef NOODLE_POOL_MODE
#define NOODLE_POOL_MODE NOODLE_POOL_NONE
#endif

#ifndef NOODLE_USE_INT8
#error "Build Noodle with -D NOODLE_USE_INT8"
#endif

#include "noodle.h"
#include "cute_model_runtime.h"

#if CUTE_YOLO_INPUT_C != 1
#error "This example requires the grayscale Cute-YOLO model."
#endif

#ifndef CUTE_YOLO_V8_REBALANCED_CNN_DW_PW
#error "This firmware requires the fixed Cute-YOLO 8+24 runtime model interface."
#endif

#ifndef CUTE_YOLO_STEM3_BRANCH_A_OUT
#error "Fixed Cute-YOLO runtime is missing split-stem metadata."
#endif

#ifndef CUTE_YOLO_HYBRID_NORMAL_OUT
#error "Fixed Cute-YOLO runtime is missing hybrid branch metadata."
#endif

#if CUTE_YOLO_HYBRID_NORMAL_OUT != 8 || CUTE_YOLO_HYBRID_EFFICIENT_OUT != 24
#error "This firmware is generated for the V8 conv8 + DW/PW24 model."
#endif

#if defined(CONFIG_FREERTOS_UNICORE) && CONFIG_FREERTOS_UNICORE
#error "Cute-YOLO V8 Rebalanced Hybrid firmware requires both ESP32-S3 CPU cores."
#endif

// ============================================================
// Cute-YOLO fixed 8+24 Hybrid — dual-core ESP32-S3
//
// Stem:
//   1 -> 8 -> 16 -> 32
//   Each stem Conv is split into two contiguous output-filter halves
//   and executed on both cores in parallel.
//
// Five heterogeneous 32-channel hybrid blocks:
//
//                       full 32-channel input
//                      /                     \
//                  Core 0                   Core 1
//             Conv3x3 32 -> 8        DW3x3 32 -> 32
//                                           |
//                                     PW1x1 32 -> 24
//                      \                     /
//                       concat 8 + 24 -> 32
//
// The trained branch order is preserved exactly: 8 normal-Conv channels
// first, followed by 24 DW/PW channels.
//
// Head:
//   32 -> 5, monolithic 1x1 Conv.
//
// Camera RGB565 -> grayscale 128x128 INT8 -> V8 Rebalanced Hybrid -> 5x16x16
// -> decode all thresholded cells -> NMS -> TFT boxes.
// ============================================================

// ------------------------------------------------------------
// TFT and button
// ------------------------------------------------------------

#define TFT_SCLK 39
#define TFT_MOSI 40
#define TFT_CS 38
#define TFT_DC 41
#define TFT_RST 42
#define TFT_MISO -1

#define BUTTON_PIN 0 // built-in BOOT button on ESP32-S3 DevKitC-1

Adafruit_ST7735 tft =
    Adafruit_ST7735(&SPI, TFT_CS, TFT_DC, TFT_RST);

// ------------------------------------------------------------
// Camera and detector geometry
// ------------------------------------------------------------

static const int PREVIEW_W = 160;
static const int PREVIEW_H = 120;

static const int DET_CROP_W = 240;
static const int DET_CROP_H = 240;

static const int GUIDE_X = 20;
static const int GUIDE_Y = 0;
static const int GUIDE_W = 120;
static const int GUIDE_H = 120;

static const uint16_t IMG_W = CUTE_YOLO_INPUT_W; // 128
static const uint16_t IMG_H = CUTE_YOLO_INPUT_H; // 128
static const uint16_t IMG_C = CUTE_YOLO_INPUT_C; // 1
static const uint16_t GRID_W = CUTE_YOLO_GRID_W; // 16

static const uint32_t GRID_PIXELS =
    (uint32_t)GRID_W * GRID_W;

// Noodle public API: P=65535 requests TensorFlow/TFLite SAME-style 2D
// padding, including asymmetric padding for stride-2 convolutions.
// This matters for the three 3x3/2 stems on even-sized feature maps.
static constexpr uint16_t CUTE_TFLITE_SAME_PADDING = 65535u;

// Detector operating point is part of the uploaded .cute model.
// Static capacities remain fixed; runtime limits are clamped to them.
static const uint8_t MAX_CANDIDATES = 128;
static const uint8_t MAX_DETECTIONS = 32;

static inline float confidence_threshold()
{
    return cute_model_confidence_threshold();
}

static inline float nms_iou_threshold()
{
    return cute_model_nms_iou_threshold();
}

static inline float min_box_w()
{
    return cute_model_min_box_w();
}

static inline float min_box_h()
{
    return cute_model_min_box_h();
}

static inline uint8_t runtime_max_detections()
{
    const uint8_t n = cute_model_max_detections();
    return n > MAX_DETECTIONS ? MAX_DETECTIONS : n;
}

static uint16_t linebuf[PREVIEW_W];

// ------------------------------------------------------------
// OPTIONAL EXACT TFT FRAME DUMP
//
// The ST7735 is write-only in this project (TFT_MISO=-1), so the
// physical panel cannot be read back. Keep a 160x128 RGB565 shadow
// framebuffer and mirror every visible overlay into it.
//
// PC heartbeat:
//     RDYDISPLAY\n
//
// After inference the firmware can send the final rendered frame:
// preview + red detector guide + boxes + optional confidence captions
// + bottom N/T status line.
// ------------------------------------------------------------

static constexpr uint16_t TFT_FRAME_W = 160;
static constexpr uint16_t TFT_FRAME_H = 128;
static constexpr size_t TFT_FRAME_PIXELS =
    (size_t)TFT_FRAME_W * TFT_FRAME_H;
static constexpr size_t TFT_FRAME_BYTES =
    TFT_FRAME_PIXELS * sizeof(uint16_t);

#ifndef CUTE_TFT_DRAW_CONFIDENCE
#define CUTE_TFT_DRAW_CONFIDENCE 1
#endif

static uint16_t *tft_shadow = nullptr;

class CuteShadowCanvas : public Adafruit_GFX
{
public:
    CuteShadowCanvas(uint16_t w, uint16_t h, uint16_t *buffer)
        : Adafruit_GFX(w, h), _buffer(buffer)
    {
    }

    void drawPixel(int16_t x, int16_t y, uint16_t color) override
    {
        if (!_buffer)
            return;

        if (x < 0 || y < 0 || x >= width() || y >= height())
            return;

        _buffer[(size_t)y * TFT_FRAME_W + (size_t)x] = color;
    }

private:
    uint16_t *_buffer;
};

static CuteShadowCanvas *tft_canvas = nullptr;

// ------------------------------------------------------------
// OPTIONAL SERIAL DATASET CAPTURE
//
// A PC collector periodically sends:
//     RDYSAMPLE\n
//
// If that heartbeat is recent, the firmware sends after inference:
//   1. the exact 128x128 grayscale image presented to Cute-YOLO,
//   2. detection coordinates/confidences as text metadata.
//
// The ESP32 never paints boxes into the transmitted training image.
// The Python collector draws a separate labeled preview and writes
// pseudo-label files on the PC.
// ------------------------------------------------------------

static constexpr size_t SAMPLE_PIXELS =
    (size_t)IMG_W * IMG_H;
static constexpr size_t SAMPLE_BYTES = SAMPLE_PIXELS;

// Heartbeat prevents binary transfer when no collector is listening.
static constexpr uint32_t SERIAL_READY_WINDOW_MS = 2500;

static uint32_t serial_last_ready_ms = 0;
static bool serial_host_seen = false;
static uint32_t serial_sample_id = 0;

static uint32_t serial_display_last_ready_ms = 0;
static bool serial_display_host_seen = false;
static uint32_t serial_display_id = 0;

static char serial_command[24];
static uint8_t serial_command_len = 0;

// Exact pre-quantization grayscale detector input, 0..255.
// Allocate in PSRAM at setup time.
static uint8_t *sample_gray = nullptr;

// ------------------------------------------------------------
// Noodle tensors
//
// X       = network input
// A/B     = full-width ping-pong tensors
// BR0     = Core-0 branch output
// BR1     = Core-1 final branch output
// DWTMP   = Core-1 depthwise intermediate
//
// Worst retained INT8 capacities:
//   X     :  1 * 128 * 128 = 16,384 B
//   A     :  8 *  64 *  64 = 32,768 B
//   B     : 16 *  32 *  32 = 16,384 B
//   BR0   :  4 *  64 *  64 = 16,384 B
//   BR1   :  4 *  64 *  64 = 16,384 B
//   DWTMP : 32 *  16 *  16 =  8,192 B
//
// Total retained tensor capacity = 106,496 bytes.
//
// All tensors are fully grown before worker creation. During inference,
// the memory-backed ConvMem/DWConv paths only reuse these allocations.
// ------------------------------------------------------------

static NoodleTensor X;
static NoodleTensor A;
static NoodleTensor B;
static NoodleTensor BR0;
static NoodleTensor BR1;
static NoodleTensor DWTMP;

// ------------------------------------------------------------
// Dual-core workers
// ------------------------------------------------------------

#define V8_WORKER_STACK_BYTES 4096
#define V8_WORKER_PRIORITY 2

#ifndef V8_PRINT_LAYER_TIMING
#define V8_PRINT_LAYER_TIMING 1
#endif

enum WorkerJobKind : uint8_t
{
    JOB_CONV = 0,
    JOB_DW_PW = 1
};

struct BranchJob
{
    WorkerJobKind kind;

    NoodleTensor *input;
    NoodleTensor *output;
    NoodleTensor *temp;

    ConvMem first;
    ConvMem second;
    Pool pool;

    volatile uint16_t result_w;
    volatile uint32_t op1_us;
    volatile uint32_t op2_us;
    volatile uint32_t elapsed_us;
};

static BranchJob branch_jobs[2];

static TaskHandle_t branch_tasks[2] = {
    nullptr,
    nullptr};

static SemaphoreHandle_t branch_done[2] = {
    nullptr,
    nullptr};

// ------------------------------------------------------------
// Detection data
// ------------------------------------------------------------

struct Detection
{
    float confidence;
    float x1;
    float y1;
    float x2;
    float y2;
};

struct DetectionSet
{
    uint8_t count;
    Detection items[MAX_DETECTIONS];
};

static Detection candidates[MAX_CANDIDATES];

// ============================================================
// Optional serial dataset transport
// ============================================================

static void poll_serial_dataset_ready()
{
    while (Serial.available() > 0)
    {
        const int c = Serial.read();
        if (c < 0)
            break;

        if (c == '\r')
        {
            continue;
        }

        if (c == '\n')
        {
            serial_command[serial_command_len] = '\0';

            if (strcmp(serial_command, "RDYSAMPLE") == 0 ||
                strcmp(serial_command, "RDYFRAME") == 0)
            {
                serial_last_ready_ms = millis();

                if (!serial_host_seen)
                {
                    serial_host_seen = true;
                    Serial.println("SERIALDATA HOST_READY");
                }
            }
            else if (strcmp(serial_command, "RDYDISPLAY") == 0)
            {
                serial_display_last_ready_ms = millis();

                if (!serial_display_host_seen)
                {
                    serial_display_host_seen = true;
                    Serial.println("DISPLAYDATA HOST_READY");
                }
            }
            else if (strcmp(serial_command, "STOPSAMPLE") == 0 ||
                     strcmp(serial_command, "STOPFRAME") == 0)
            {
                serial_last_ready_ms = 0;
                serial_host_seen = false;
                Serial.println("SERIALDATA STOPPED");
            }
            else if (strcmp(serial_command, "STOPDISPLAY") == 0)
            {
                serial_display_last_ready_ms = 0;
                serial_display_host_seen = false;
                Serial.println("DISPLAYDATA STOPPED");
            }

            serial_command_len = 0;
            continue;
        }

        if (serial_command_len + 1 < sizeof(serial_command))
        {
            serial_command[serial_command_len++] = (char)c;
        }
        else
        {
            serial_command_len = 0;
        }
    }
}

static bool serial_dataset_receiver_ready()
{
    if (serial_last_ready_ms == 0)
    {
        return false;
    }

    return (uint32_t)(millis() - serial_last_ready_ms) <= SERIAL_READY_WINDOW_MS;
}

static bool serial_display_receiver_ready()
{
    if (serial_display_last_ready_ms == 0)
    {
        return false;
    }

    return (uint32_t)(millis() - serial_display_last_ready_ms) <=
           SERIAL_READY_WINDOW_MS;
}

static bool init_sample_gray()
{
    if (sample_gray)
    {
        return true;
    }

#if defined(ARDUINO_ARCH_ESP32)
    if (psramFound())
    {
        sample_gray =
            (uint8_t *)ps_malloc(SAMPLE_BYTES);
    }
#endif

    if (!sample_gray)
    {
        sample_gray =
            (uint8_t *)malloc(SAMPLE_BYTES);
    }

    if (!sample_gray)
    {
        return false;
    }

    memset(sample_gray, 0, SAMPLE_BYTES);
    return true;
}

static bool serial_write_binary(
    const uint8_t *data,
    size_t n)
{

    if (!data)
        return false;

    size_t sent = 0;

    while (sent < n)
    {
        const size_t chunk =
            min((size_t)64, n - sent);

        const size_t wrote =
            Serial.write(data + sent, chunk);

        if (wrote > 0)
        {
            sent += wrote;
        }
        else
        {
            delay(1);
        }

        // Keep USB CDC stable on the S3.
        delay(1);
    }

    Serial.flush();
    return true;
}

static void send_dataset_sample_if_ready(
    const DetectionSet &detections,
    uint32_t inference_us)
{

    // Consume heartbeat that may have arrived while inference was running.
    poll_serial_dataset_ready();

    if (!serial_dataset_receiver_ready())
    {
        return;
    }

    if (!sample_gray)
    {
        Serial.println("ERR_SAMPLE_GRAY_NULL");
        return;
    }

    const uint32_t id = serial_sample_id++;
    const char *label = cute_model_label();

    Serial.printf(
        "SAMPLE %lu %s %lu %u\n",
        (unsigned long)id,
        label,
        (unsigned long)inference_us,
        (unsigned)detections.count);

    Serial.printf(
        "RAW %u %u GRAY8 %u\n",
        (unsigned)IMG_W,
        (unsigned)IMG_H,
        (unsigned)SAMPLE_BYTES);

    Serial.flush();

    if (!serial_write_binary(
            sample_gray,
            SAMPLE_BYTES))
    {
        Serial.println("ERR_SAMPLE_WRITE");
        return;
    }

    // Delimiter after the fixed-size binary block.
    Serial.print('\n');

    Serial.printf(
        "DET %u\n",
        (unsigned)detections.count);

    for (uint8_t i = 0;
         i < detections.count;
         ++i)
    {

        const Detection &d =
            detections.items[i];

        Serial.printf(
            "BOX %.6f %.6f %.6f %.6f %.6f\n",
            (double)d.confidence,
            (double)d.x1,
            (double)d.y1,
            (double)d.x2,
            (double)d.y2);
    }

    Serial.println("END");
    NoodleSerial::print_ready();
    Serial.flush();
}

static void send_display_frame_if_ready(
    const DetectionSet &detections,
    uint32_t inference_us)
{
    poll_serial_dataset_ready();

    if (!serial_display_receiver_ready())
    {
        return;
    }

    if (!tft_shadow)
    {
        Serial.println("ERR_DISPLAY_SHADOW_NULL");
        return;
    }

    const uint32_t id = serial_display_id++;
    const char *label = cute_model_label();

    Serial.printf(
        "DISPLAY %lu %s %lu %u\n",
        (unsigned long)id,
        label,
        (unsigned long)inference_us,
        (unsigned)detections.count);

    Serial.printf(
        "FRAME %u %u RGB565LE %u\n",
        (unsigned)TFT_FRAME_W,
        (unsigned)TFT_FRAME_H,
        (unsigned)TFT_FRAME_BYTES);

    Serial.flush();

    if (!serial_write_binary(
            reinterpret_cast<const uint8_t *>(tft_shadow),
            TFT_FRAME_BYTES))
    {
        Serial.println("ERR_DISPLAY_WRITE");
        return;
    }

    Serial.print('\n');

    Serial.printf(
        "DET %u\n",
        (unsigned)detections.count);

    for (uint8_t i = 0; i < detections.count; ++i)
    {
        const Detection &d = detections.items[i];

        Serial.printf(
            "BOX %.6f %.6f %.6f %.6f %.6f\n",
            (double)d.confidence,
            (double)d.x1,
            (double)d.y1,
            (double)d.x2,
            (double)d.y2);
    }

    Serial.println("ENDDISPLAY");
    NoodleSerial::print_ready();
    Serial.flush();
}

// ============================================================
// Small helpers
// ============================================================

static inline uint16_t swap565(uint16_t c)
{
    return (uint16_t)((c >> 8) | (c << 8));
}

static inline float clamp01(float x)
{
    if (x < 0.0f)
        return 0.0f;
    if (x > 1.0f)
        return 1.0f;
    return x;
}

static inline float sigmoid(float x)
{
    return 1.0f / (1.0f + expf(-x));
}

static void status(const char *text,
                   uint16_t color = ST77XX_GREEN)
{
    tft.fillRect(0, 120, 160, 8, ST77XX_BLACK);
    tft.setCursor(2, 121);
    tft.setTextSize(1);
    tft.setTextColor(color, ST77XX_BLACK);
    tft.print(text);

    if (tft_canvas)
    {
        tft_canvas->fillRect(0, 120, 160, 8, ST77XX_BLACK);
        tft_canvas->setCursor(2, 121);
        tft_canvas->setTextSize(1);
        tft_canvas->setTextColor(color, ST77XX_BLACK);
        tft_canvas->print(text);
    }
}

// ============================================================
// Camera and TFT
// ============================================================

static bool init_camera()
{
    camera_config_t config;
    cam_fill_pins(config);

    // Keep RGB565 so the TFT preview remains in color.
    // Only the neural-network input is converted to grayscale.
    config.pixel_format = PIXFORMAT_RGB565;
    config.frame_size = FRAMESIZE_QVGA;
    config.fb_count = 1;
    config.fb_location = CAMERA_FB_IN_PSRAM;
    config.grab_mode = CAMERA_GRAB_WHEN_EMPTY;

    if (esp_camera_init(&config) != ESP_OK)
    {
        return false;
    }

    sensor_t *sensor = esp_camera_sensor_get();
    if (sensor)
    {
        // Darker camera settings already validated on this target board.
        sensor->set_exposure_ctrl(sensor, 1);
        sensor->set_gain_ctrl(sensor, 1);
        sensor->set_ae_level(sensor, -2);
        sensor->set_brightness(sensor, -2);
        sensor->set_contrast(sensor, 0);
        sensor->set_saturation(sensor, 0);
        sensor->set_hmirror(sensor, 0);
        sensor->set_vflip(sensor, 1);
    }

    return true;
}

static bool init_tft_shadow()
{
    if (tft_shadow && tft_canvas)
    {
        return true;
    }

#if defined(ARDUINO_ARCH_ESP32)
    if (psramFound())
    {
        tft_shadow =
            (uint16_t *)ps_malloc(TFT_FRAME_BYTES);
    }
#endif

    if (!tft_shadow)
    {
        tft_shadow =
            (uint16_t *)malloc(TFT_FRAME_BYTES);
    }

    if (!tft_shadow)
    {
        return false;
    }

    memset(tft_shadow, 0, TFT_FRAME_BYTES);

    tft_canvas =
        new CuteShadowCanvas(
            TFT_FRAME_W,
            TFT_FRAME_H,
            tft_shadow);

    if (!tft_canvas)
    {
        free(tft_shadow);
        tft_shadow = nullptr;
        return false;
    }

    tft_canvas->setTextWrap(false);
    tft_canvas->setTextSize(1);
    tft_canvas->setTextColor(ST77XX_WHITE, ST77XX_BLACK);

    return true;
}

static void init_tft()
{
    SPI.begin(TFT_SCLK, TFT_MISO, TFT_MOSI, TFT_CS);

    tft.initR(INITR_BLACKTAB);
    tft.setRotation(1);
    tft.fillScreen(ST77XX_BLACK);
    tft.setTextWrap(false);
    tft.setTextSize(1);
}

// ============================================================
// Live color preview
// ============================================================

static void draw_preview(const camera_fb_t *fb)
{
    const uint16_t *src = (const uint16_t *)fb->buf;
    const int src_w = fb->width;

    // QVGA 320x240 -> TFT 160x120.
    for (int y = 0; y < PREVIEW_H; ++y)
    {
        const uint16_t *row = src + (y * 2) * src_w;

        for (int x = 0; x < PREVIEW_W; ++x)
        {
            linebuf[x] = swap565(row[x * 2]);
        }

        tft.drawRGBBitmap(
            0, y,
            linebuf,
            PREVIEW_W, 1);

        if (tft_shadow)
        {
            memcpy(
                tft_shadow + (size_t)y * TFT_FRAME_W,
                linebuf,
                PREVIEW_W * sizeof(uint16_t));
        }
    }

    // Red rectangle = detector crop.
    tft.drawRect(
        GUIDE_X, GUIDE_Y,
        GUIDE_W, GUIDE_H,
        ST77XX_RED);

    if (tft_canvas)
    {
        tft_canvas->drawRect(
            GUIDE_X, GUIDE_Y,
            GUIDE_W, GUIDE_H,
            ST77XX_RED);
    }
}

// ============================================================
// RGB565 -> grayscale
// ============================================================

static inline float rgb565_to_gray(uint16_t raw)
{
    const uint16_t p = swap565(raw);

    const float r =
        (float)((p >> 11) & 0x1F) / 31.0f;
    const float g =
        (float)((p >> 5) & 0x3F) / 63.0f;
    const float b =
        (float)(p & 0x1F) / 31.0f;

    return 0.299f * r +
           0.587f * g +
           0.114f * b;
}

// ============================================================
// Camera frame -> 1x128x128 INT8 tensor
// ============================================================

static bool frame_to_tensor(const camera_fb_t *fb)
{
    if (!fb || fb->format != PIXFORMAT_RGB565)
    {
        return false;
    }

    noodle_tensor_set_quantization(
        &X,
        CUTE_YOLO_INPUT_SCALE,
        CUTE_YOLO_INPUT_ZERO_POINT);

    NoodleData *x =
        noodle_tensor_require_2d(&X, IMG_C, IMG_W);

    if (!x)
        return false;

    const int src_w = fb->width;
    const int src_h = fb->height;

    const int crop_x =
        (src_w - DET_CROP_W) / 2;
    const int crop_y =
        (src_h - DET_CROP_H) / 2;

    const uint16_t *src =
        (const uint16_t *)fb->buf;

    // Bilinear resize 240x240 -> 128x128.
    for (uint16_t y = 0; y < IMG_H; ++y)
    {
        float sy =
            ((float)y + 0.5f) *
                ((float)DET_CROP_H / IMG_H) -
            0.5f;

        sy = fmaxf(0.0f, fminf(
                             sy, DET_CROP_H - 1.0f));

        const int y0l = (int)floorf(sy);
        const int y1l =
            min(y0l + 1, DET_CROP_H - 1);

        const float wy = sy - y0l;

        const int y0 = crop_y + y0l;
        const int y1 = crop_y + y1l;

        for (uint16_t xp = 0; xp < IMG_W; ++xp)
        {
            float sx =
                ((float)xp + 0.5f) *
                    ((float)DET_CROP_W / IMG_W) -
                0.5f;

            sx = fmaxf(0.0f, fminf(
                                 sx, DET_CROP_W - 1.0f));

            const int x0l = (int)floorf(sx);
            const int x1l =
                min(x0l + 1, DET_CROP_W - 1);

            const float wx = sx - x0l;

            const int x0 = crop_x + x0l;
            const int x1 = crop_x + x1l;

            const float g00 =
                rgb565_to_gray(src[y0 * src_w + x0]);
            const float g10 =
                rgb565_to_gray(src[y0 * src_w + x1]);
            const float g01 =
                rgb565_to_gray(src[y1 * src_w + x0]);
            const float g11 =
                rgb565_to_gray(src[y1 * src_w + x1]);

            const float top =
                g00 + wx * (g10 - g00);
            const float bottom =
                g01 + wx * (g11 - g01);

            const float gray =
                top + wy * (bottom - top);

            const uint32_t i =
                (uint32_t)y * IMG_W + xp;

            // Preserve the exact real-valued detector input as GRAY8 for
            // dataset collection. This is BEFORE INT8 model quantization.
            if (sample_gray)
            {
                const int gray_u8 =
                    (int)lroundf(gray * 255.0f);

                sample_gray[i] =
                    (uint8_t)constrain(gray_u8, 0, 255);
            }

            x[i] = (NoodleData)noodle_quantize_float(
                gray,
                CUTE_YOLO_INPUT_SCALE,
                CUTE_YOLO_INPUT_ZERO_POINT);
        }
    }

    return true;
}

// ============================================================
// Cute-YOLO model
// ============================================================

static void make_conv(
    ConvMem &conv,
    uint16_t K,
    uint16_t P,
    uint16_t S,
    uint16_t O,
    const NoodleWeight *weight,
    const NoodleBias *bias,
    const int32_t *multiplier,
    const int32_t *shift,
    float input_scale,
    int32_t input_zero_point,
    float output_scale,
    int32_t output_zero_point,
    Activation activation)
{

    conv.K = K;
    conv.P = P;
    conv.S = S;
    conv.OP = 0;
    conv.O = O;

    conv.weight = weight;
    conv.bias = bias;

    conv.multiplier = multiplier;
    conv.shift = shift;

    conv.input_scale = input_scale;
    conv.input_zero_point = input_zero_point;

    conv.output_scale = output_scale;
    conv.output_zero_point = output_zero_point;

    conv.activation_min = -128;
    conv.activation_max = 127;
    conv.depth_multiplier = 1;
    conv.act = activation;
}

static Pool no_pool()
{
    Pool p;
    p.M = 1;
    p.T = 1;
    return p;
}

// ============================================================
// Persistent dual-core workers
// ============================================================
//
// Current Noodle memory-backed INT8 ConvMem and depthwise kernels do not
// use the file-backed shared filter scratch. The two workers therefore:
//   - read one immutable full input tensor,
//   - read disjoint immutable parameter arrays,
//   - write distinct preallocated tensors.
//
// Core 0 always runs a normal Conv branch.
// Core 1 runs either a normal Conv branch (stem) or DW -> PW (hybrid).
// ============================================================

static void branch_worker(void *arg)
{
    const int worker =
        (int)(intptr_t)arg;

    for (;;)
    {
        ulTaskNotifyTake(
            pdTRUE,
            portMAX_DELAY);

        BranchJob &job =
            branch_jobs[worker];

        job.result_w = 0;
        job.op1_us = 0;
        job.op2_us = 0;
        job.elapsed_us = 0;

        const uint32_t total_t0 =
            micros();

        if (job.kind == JOB_CONV)
        {
            const uint32_t t0 =
                micros();

            job.result_w =
                noodle_conv2d(
                    job.input,
                    job.output,
                    job.first,
                    job.pool);

            job.op1_us =
                micros() - t0;
        }
        else
        {
            const uint32_t dw_t0 =
                micros();

            const uint16_t dw_w =
                noodle_dwconv2d(
                    job.input,
                    job.temp,
                    job.first,
                    job.pool);

            job.op1_us =
                micros() - dw_t0;

            if (dw_w)
            {
                const uint32_t pw_t0 =
                    micros();

                job.result_w =
                    noodle_conv2d(
                        job.temp,
                        job.output,
                        job.second,
                        job.pool);

                job.op2_us =
                    micros() - pw_t0;
            }
        }

        job.elapsed_us =
            micros() - total_t0;

        xSemaphoreGive(
            branch_done[worker]);
    }
}

static bool init_dualcore_workers()
{
    branch_done[0] =
        xSemaphoreCreateBinary();

    branch_done[1] =
        xSemaphoreCreateBinary();

    if (!branch_done[0] ||
        !branch_done[1])
    {
        return false;
    }

    const BaseType_t ok0 =
        xTaskCreatePinnedToCore(
            branch_worker,
            "v6_core0",
            V8_WORKER_STACK_BYTES,
            (void *)0,
            V8_WORKER_PRIORITY,
            &branch_tasks[0],
            0);

    const BaseType_t ok1 =
        xTaskCreatePinnedToCore(
            branch_worker,
            "v6_core1",
            V8_WORKER_STACK_BYTES,
            (void *)1,
            V8_WORKER_PRIORITY,
            &branch_tasks[1],
            1);

    return ok0 == pdPASS &&
           ok1 == pdPASS;
}

static void clear_worker_tokens()
{
    xSemaphoreTake(
        branch_done[0],
        0);

    xSemaphoreTake(
        branch_done[1],
        0);
}

static bool launch_and_wait(
    uint32_t *parallel_wall_us)
{

    const uint32_t wall_t0 =
        micros();

    xTaskNotifyGive(
        branch_tasks[0]);

    xTaskNotifyGive(
        branch_tasks[1]);

    xSemaphoreTake(
        branch_done[0],
        portMAX_DELAY);

    xSemaphoreTake(
        branch_done[1],
        portMAX_DELAY);

    if (parallel_wall_us)
    {
        *parallel_wall_us =
            micros() - wall_t0;
    }

    const uint16_t w0 =
        branch_jobs[0].result_w;

    const uint16_t w1 =
        branch_jobs[1].result_w;

    return w0 != 0 &&
           w1 != 0 &&
           w0 == w1;
}

static bool concat_branches(
    const char *name,
    NoodleTensor *output,
    uint16_t expected_c,
    uint32_t *concat_us)
{

    const uint32_t t0 =
        micros();

    const uint16_t combined_c =
        noodle_concat(
            &BR0,
            &BR1,
            output);

    const uint32_t elapsed =
        micros() - t0;

    if (concat_us)
    {
        *concat_us =
            elapsed;
    }

    if (combined_c != expected_c)
    {
#if V8_PRINT_LAYER_TIMING
        Serial.printf(
            "%s CONCAT FAIL C=%u expected=%u "
            "BR0(s=%.9g,zp=%ld) BR1(s=%.9g,zp=%ld)\n",
            name,
            (unsigned)combined_c,
            (unsigned)expected_c,
            (double)BR0.scale,
            (long)BR0.zero_point,
            (double)BR1.scale,
            (long)BR1.zero_point);
#endif
        return false;
    }

    return true;
}

static bool run_split_conv_layer(
    const char *name,
    NoodleTensor *input,
    NoodleTensor *output,
    const ConvMem &conv_a,
    const ConvMem &conv_b)
{

    if (!input ||
        !output ||
        !branch_tasks[0] ||
        !branch_tasks[1])
    {
        return false;
    }

    clear_worker_tokens();

    const Pool p =
        no_pool();

    branch_jobs[0].kind =
        JOB_CONV;
    branch_jobs[0].input =
        input;
    branch_jobs[0].output =
        &BR0;
    branch_jobs[0].temp =
        nullptr;
    branch_jobs[0].first =
        conv_a;
    branch_jobs[0].second =
        ConvMem();
    branch_jobs[0].pool =
        p;

    branch_jobs[1].kind =
        JOB_CONV;
    branch_jobs[1].input =
        input;
    branch_jobs[1].output =
        &BR1;
    branch_jobs[1].temp =
        nullptr;
    branch_jobs[1].first =
        conv_b;
    branch_jobs[1].second =
        ConvMem();
    branch_jobs[1].pool =
        p;

    uint32_t parallel_us = 0;

    if (!launch_and_wait(
            &parallel_us))
    {

#if V8_PRINT_LAYER_TIMING
        Serial.printf(
            "%s FAIL A_W=%u B_W=%u\n",
            name,
            (unsigned)branch_jobs[0].result_w,
            (unsigned)branch_jobs[1].result_w);
#endif

        return false;
    }

    uint32_t concat_us = 0;

    if (!concat_branches(
            name,
            output,
            (uint16_t)(conv_a.O +
                       conv_b.O),
            &concat_us))
    {
        return false;
    }

#if V8_PRINT_LAYER_TIMING
    Serial.printf(
        "%s  A=%luus B=%luus "
        "parallel=%luus concat=%luus "
        "-> %ux%ux%u\n",
        name,
        (unsigned long)
            branch_jobs[0]
                .elapsed_us,
        (unsigned long)
            branch_jobs[1]
                .elapsed_us,
        (unsigned long)
            parallel_us,
        (unsigned long)
            concat_us,
        (unsigned)output->C,
        (unsigned)output->W,
        (unsigned)output->W);
#endif

    return true;
}

static bool run_hybrid_layer(
    const char *name,
    NoodleTensor *input,
    NoodleTensor *output,
    const ConvMem &normal_conv,
    const ConvMem &dw_conv,
    const ConvMem &pw_conv)
{

    if (!input ||
        !output ||
        !branch_tasks[0] ||
        !branch_tasks[1])
    {
        return false;
    }

    clear_worker_tokens();

    const Pool p =
        no_pool();

    // Core 0: normal 3x3 Conv 32 -> 8.
    branch_jobs[0].kind =
        JOB_CONV;
    branch_jobs[0].input =
        input;
    branch_jobs[0].output =
        &BR0;
    branch_jobs[0].temp =
        nullptr;
    branch_jobs[0].first =
        normal_conv;
    branch_jobs[0].second =
        ConvMem();
    branch_jobs[0].pool =
        p;

    // Core 1: DW3x3 32 -> 32, then PW1x1 32 -> 24.
    branch_jobs[1].kind =
        JOB_DW_PW;
    branch_jobs[1].input =
        input;
    branch_jobs[1].output =
        &BR1;
    branch_jobs[1].temp =
        &DWTMP;
    branch_jobs[1].first =
        dw_conv;
    branch_jobs[1].second =
        pw_conv;
    branch_jobs[1].pool =
        p;

    uint32_t parallel_us = 0;

    if (!launch_and_wait(
            &parallel_us))
    {

#if V8_PRINT_LAYER_TIMING
        Serial.printf(
            "%s FAIL normal_W=%u dwpw_W=%u\n",
            name,
            (unsigned)branch_jobs[0].result_w,
            (unsigned)branch_jobs[1].result_w);
#endif

        return false;
    }

    uint32_t concat_us = 0;

    if (!concat_branches(
            name,
            output,
            (uint16_t)(normal_conv.O +
                       pw_conv.O),
            &concat_us))
    {
        return false;
    }

#if V8_PRINT_LAYER_TIMING
    Serial.printf(
        "%s  Conv=%luus "
        "DW=%luus PW=%luus DWPW=%luus "
        "parallel=%luus concat=%luus "
        "-> %ux%ux%u\n",
        name,
        (unsigned long)
            branch_jobs[0]
                .elapsed_us,
        (unsigned long)
            branch_jobs[1]
                .op1_us,
        (unsigned long)
            branch_jobs[1]
                .op2_us,
        (unsigned long)
            branch_jobs[1]
                .elapsed_us,
        (unsigned long)
            parallel_us,
        (unsigned long)
            concat_us,
        (unsigned)output->C,
        (unsigned)output->W,
        (unsigned)output->W);
#endif

    return true;
}

// ============================================================
// Detection helpers
// ============================================================

static float tensor_value(
    const NoodleTensor *tensor,
    NoodleData q)
{

    return noodle_dequantize_int8(
        (int8_t)q,
        tensor->scale,
        tensor->zero_point);
}

static float iou(
    const Detection &a,
    const Detection &b)
{

    const float x1 = fmaxf(a.x1, b.x1);
    const float y1 = fmaxf(a.y1, b.y1);
    const float x2 = fminf(a.x2, b.x2);
    const float y2 = fminf(a.y2, b.y2);

    const float inter =
        fmaxf(0.0f, x2 - x1) *
        fmaxf(0.0f, y2 - y1);

    const float area_a =
        (a.x2 - a.x1) * (a.y2 - a.y1);

    const float area_b =
        (b.x2 - b.x1) * (b.y2 - b.y1);

    return inter /
           (area_a + area_b - inter + 1e-9f);
}

static void add_candidate(
    const Detection &d,
    uint8_t &count)
{

    if (count < MAX_CANDIDATES)
    {
        candidates[count++] = d;
        return;
    }

    uint8_t weakest = 0;

    for (uint8_t i = 1; i < count; ++i)
    {
        if (candidates[i].confidence <
            candidates[weakest].confidence)
        {
            weakest = i;
        }
    }

    if (d.confidence >
        candidates[weakest].confidence)
    {
        candidates[weakest] = d;
    }
}

static void sort_candidates(uint8_t count)
{
    for (uint8_t i = 1; i < count; ++i)
    {
        Detection key = candidates[i];
        int j = i - 1;

        while (j >= 0 &&
               candidates[j].confidence <
                   key.confidence)
        {
            candidates[j + 1] = candidates[j];
            --j;
        }

        candidates[j + 1] = key;
    }
}

// ============================================================
// Decode the 5x16x16 output
// ============================================================

static bool decode(
    NoodleTensor *output,
    DetectionSet *result)
{

    const NoodleData *out =
        noodle_tensor_const_data(output);

    if (!out)
        return false;

    uint8_t candidate_count = 0;

    for (int gy = 0; gy < GRID_W; ++gy)
    {
        for (int gx = 0; gx < GRID_W; ++gx)
        {
            const uint32_t i =
                (uint32_t)gy * GRID_W + gx;

            const NoodleData object_q =
                out[0 * GRID_PIXELS + i];

            // Cute-YOLO fixed detector:
            // Do NOT suppress neighboring objectness peaks here.
            // Every cell above the confidence threshold is decoded first;
            // duplicate boxes are handled later by NMS.

            const float confidence =
                sigmoid(tensor_value(output, object_q));

            if (confidence < confidence_threshold())
            {
                continue;
            }

            const float dx =
                sigmoid(tensor_value(
                    output,
                    out[1 * GRID_PIXELS + i]));

            const float dy =
                sigmoid(tensor_value(
                    output,
                    out[2 * GRID_PIXELS + i]));

            const float w =
                sigmoid(tensor_value(
                    output,
                    out[3 * GRID_PIXELS + i]));

            const float h =
                sigmoid(tensor_value(
                    output,
                    out[4 * GRID_PIXELS + i]));

            // Reject implausibly small detections.
            if (w < min_box_w() || h < min_box_h())
            {
                continue;
            }

            const float cx =
                ((float)gx + dx) / GRID_W;

            const float cy =
                ((float)gy + dy) / GRID_W;

            Detection d;

            d.confidence = confidence;
            d.x1 = clamp01(cx - 0.5f * w);
            d.y1 = clamp01(cy - 0.5f * h);
            d.x2 = clamp01(cx + 0.5f * w);
            d.y2 = clamp01(cy + 0.5f * h);

            add_candidate(d, candidate_count);
        }
    }

    // Non-maximum suppression.
    sort_candidates(candidate_count);

    result->count = 0;

    for (uint8_t i = 0;
         i < candidate_count &&
         result->count < runtime_max_detections();
         ++i)
    {

        bool suppress = false;

        for (uint8_t j = 0;
             j < result->count;
             ++j)
        {

            if (iou(candidates[i],
                    result->items[j]) >=
                nms_iou_threshold())
            {

                suppress = true;
                break;
            }
        }

        if (!suppress)
        {
            result->items[result->count++] =
                candidates[i];
        }
    }

    return true;
}

// ============================================================
// Run fixed Cute-YOLO 8+24 Hybrid
// ============================================================

static bool predict(DetectionSet *result)
{
    const Pool p =
        no_pool();

    // Stem: split output filters across both cores.
    //
    // IMPORTANT: TensorFlow Conv2D(padding="same", stride=2) on even input
    // sizes uses asymmetric SAME padding (top/left=0, bottom/right=1).
    // Noodle's explicit P=1 is symmetric and shifts the receptive-field grid.
    // Therefore the three stride-2 stems use P=65535 (TFLite SAME mode).
    ConvMem s1a, s1b;
    ConvMem s2a, s2b;
    ConvMem s3a, s3b;

    // Hybrid blocks:
    //   A  = normal Conv3x3 32->8
    //   DW = depthwise Conv3x3 32->32
    //   PW = pointwise Conv1x1 32->24
    ConvMem h1a, h1dw, h1pw;
    ConvMem h2a, h2dw, h2pw;
    ConvMem h3a, h3dw, h3pw;
    ConvMem h4a, h4dw, h4pw;
    ConvMem h5a, h5dw, h5pw;

    ConvMem head;

    // ----------------------------------------------------------
    // Stem 1: 1x128x128 -> (4+4)x64x64 -> 8x64x64
    // ----------------------------------------------------------

    make_conv(
        s1a, 3, CUTE_TFLITE_SAME_PADDING, 2,
        CUTE_YOLO_STEM1_BRANCH_A_OUT,
        w01a, b01a, m01a, s01a,
        CUTE_YOLO_INPUT_SCALE,
        CUTE_YOLO_INPUT_ZERO_POINT,
        CUTE_YOLO_STEM1_OUTPUT_SCALE,
        CUTE_YOLO_STEM1_OUTPUT_ZERO_POINT,
        ACT_RELU);

    make_conv(
        s1b, 3, CUTE_TFLITE_SAME_PADDING, 2,
        CUTE_YOLO_STEM1_BRANCH_B_OUT,
        w01b, b01b, m01b, s01b,
        CUTE_YOLO_INPUT_SCALE,
        CUTE_YOLO_INPUT_ZERO_POINT,
        CUTE_YOLO_STEM1_OUTPUT_SCALE,
        CUTE_YOLO_STEM1_OUTPUT_ZERO_POINT,
        ACT_RELU);

    // ----------------------------------------------------------
    // Stem 2: 8x64x64 -> (8+8)x32x32 -> 16x32x32
    // ----------------------------------------------------------

    make_conv(
        s2a, 3, CUTE_TFLITE_SAME_PADDING, 2,
        CUTE_YOLO_STEM2_BRANCH_A_OUT,
        w02a, b02a, m02a, s02a,
        CUTE_YOLO_STEM1_OUTPUT_SCALE,
        CUTE_YOLO_STEM1_OUTPUT_ZERO_POINT,
        CUTE_YOLO_STEM2_OUTPUT_SCALE,
        CUTE_YOLO_STEM2_OUTPUT_ZERO_POINT,
        ACT_RELU);

    make_conv(
        s2b, 3, CUTE_TFLITE_SAME_PADDING, 2,
        CUTE_YOLO_STEM2_BRANCH_B_OUT,
        w02b, b02b, m02b, s02b,
        CUTE_YOLO_STEM1_OUTPUT_SCALE,
        CUTE_YOLO_STEM1_OUTPUT_ZERO_POINT,
        CUTE_YOLO_STEM2_OUTPUT_SCALE,
        CUTE_YOLO_STEM2_OUTPUT_ZERO_POINT,
        ACT_RELU);

    // ----------------------------------------------------------
    // Stem 3: 16x32x32 -> (16+16)x16x16 -> 32x16x16
    // ----------------------------------------------------------

    make_conv(
        s3a, 3, CUTE_TFLITE_SAME_PADDING, 2,
        CUTE_YOLO_STEM3_BRANCH_A_OUT,
        w03a, b03a, m03a, s03a,
        CUTE_YOLO_STEM2_OUTPUT_SCALE,
        CUTE_YOLO_STEM2_OUTPUT_ZERO_POINT,
        CUTE_YOLO_STEM3_OUTPUT_SCALE,
        CUTE_YOLO_STEM3_OUTPUT_ZERO_POINT,
        ACT_RELU);

    make_conv(
        s3b, 3, CUTE_TFLITE_SAME_PADDING, 2,
        CUTE_YOLO_STEM3_BRANCH_B_OUT,
        w03b, b03b, m03b, s03b,
        CUTE_YOLO_STEM2_OUTPUT_SCALE,
        CUTE_YOLO_STEM2_OUTPUT_ZERO_POINT,
        CUTE_YOLO_STEM3_OUTPUT_SCALE,
        CUTE_YOLO_STEM3_OUTPUT_ZERO_POINT,
        ACT_RELU);

    // ----------------------------------------------------------
    // Hybrid 1
    // ----------------------------------------------------------

    make_conv(
        h1a, 3, 1, 1,
        CUTE_YOLO_HYBRID_NORMAL_OUT,
        w_h1a, b_h1a, m_h1a, s_h1a,
        CUTE_YOLO_STEM3_OUTPUT_SCALE,
        CUTE_YOLO_STEM3_OUTPUT_ZERO_POINT,
        CUTE_YOLO_H1_OUTPUT_SCALE,
        CUTE_YOLO_H1_OUTPUT_ZERO_POINT,
        ACT_RELU);

    make_conv(
        h1dw, 3, 1, 1,
        32,
        w_h1dw, b_h1dw, m_h1dw, s_h1dw,
        CUTE_YOLO_STEM3_OUTPUT_SCALE,
        CUTE_YOLO_STEM3_OUTPUT_ZERO_POINT,
        CUTE_YOLO_H1_DW_OUTPUT_SCALE,
        CUTE_YOLO_H1_DW_OUTPUT_ZERO_POINT,
        ACT_RELU);

    h1dw.depth_multiplier = 1;

    make_conv(
        h1pw, 1, 0, 1,
        CUTE_YOLO_HYBRID_EFFICIENT_OUT,
        w_h1pw, b_h1pw, m_h1pw, s_h1pw,
        CUTE_YOLO_H1_DW_OUTPUT_SCALE,
        CUTE_YOLO_H1_DW_OUTPUT_ZERO_POINT,
        CUTE_YOLO_H1_OUTPUT_SCALE,
        CUTE_YOLO_H1_OUTPUT_ZERO_POINT,
        ACT_RELU);

    // ----------------------------------------------------------
    // Hybrid 2
    // ----------------------------------------------------------

    make_conv(
        h2a, 3, 1, 1,
        CUTE_YOLO_HYBRID_NORMAL_OUT,
        w_h2a, b_h2a, m_h2a, s_h2a,
        CUTE_YOLO_H1_OUTPUT_SCALE,
        CUTE_YOLO_H1_OUTPUT_ZERO_POINT,
        CUTE_YOLO_H2_OUTPUT_SCALE,
        CUTE_YOLO_H2_OUTPUT_ZERO_POINT,
        ACT_RELU);

    make_conv(
        h2dw, 3, 1, 1,
        32,
        w_h2dw, b_h2dw, m_h2dw, s_h2dw,
        CUTE_YOLO_H1_OUTPUT_SCALE,
        CUTE_YOLO_H1_OUTPUT_ZERO_POINT,
        CUTE_YOLO_H2_DW_OUTPUT_SCALE,
        CUTE_YOLO_H2_DW_OUTPUT_ZERO_POINT,
        ACT_RELU);

    h2dw.depth_multiplier = 1;

    make_conv(
        h2pw, 1, 0, 1,
        CUTE_YOLO_HYBRID_EFFICIENT_OUT,
        w_h2pw, b_h2pw, m_h2pw, s_h2pw,
        CUTE_YOLO_H2_DW_OUTPUT_SCALE,
        CUTE_YOLO_H2_DW_OUTPUT_ZERO_POINT,
        CUTE_YOLO_H2_OUTPUT_SCALE,
        CUTE_YOLO_H2_OUTPUT_ZERO_POINT,
        ACT_RELU);

    // ----------------------------------------------------------
    // Hybrid 3
    // ----------------------------------------------------------

    make_conv(
        h3a, 3, 1, 1,
        CUTE_YOLO_HYBRID_NORMAL_OUT,
        w_h3a, b_h3a, m_h3a, s_h3a,
        CUTE_YOLO_H2_OUTPUT_SCALE,
        CUTE_YOLO_H2_OUTPUT_ZERO_POINT,
        CUTE_YOLO_H3_OUTPUT_SCALE,
        CUTE_YOLO_H3_OUTPUT_ZERO_POINT,
        ACT_RELU);

    make_conv(
        h3dw, 3, 1, 1,
        32,
        w_h3dw, b_h3dw, m_h3dw, s_h3dw,
        CUTE_YOLO_H2_OUTPUT_SCALE,
        CUTE_YOLO_H2_OUTPUT_ZERO_POINT,
        CUTE_YOLO_H3_DW_OUTPUT_SCALE,
        CUTE_YOLO_H3_DW_OUTPUT_ZERO_POINT,
        ACT_RELU);

    h3dw.depth_multiplier = 1;

    make_conv(
        h3pw, 1, 0, 1,
        CUTE_YOLO_HYBRID_EFFICIENT_OUT,
        w_h3pw, b_h3pw, m_h3pw, s_h3pw,
        CUTE_YOLO_H3_DW_OUTPUT_SCALE,
        CUTE_YOLO_H3_DW_OUTPUT_ZERO_POINT,
        CUTE_YOLO_H3_OUTPUT_SCALE,
        CUTE_YOLO_H3_OUTPUT_ZERO_POINT,
        ACT_RELU);

    // ----------------------------------------------------------
    // Hybrid 4
    // ----------------------------------------------------------

    make_conv(
        h4a, 3, 1, 1,
        CUTE_YOLO_HYBRID_NORMAL_OUT,
        w_h4a, b_h4a, m_h4a, s_h4a,
        CUTE_YOLO_H3_OUTPUT_SCALE,
        CUTE_YOLO_H3_OUTPUT_ZERO_POINT,
        CUTE_YOLO_H4_OUTPUT_SCALE,
        CUTE_YOLO_H4_OUTPUT_ZERO_POINT,
        ACT_RELU);

    make_conv(
        h4dw, 3, 1, 1,
        32,
        w_h4dw, b_h4dw, m_h4dw, s_h4dw,
        CUTE_YOLO_H3_OUTPUT_SCALE,
        CUTE_YOLO_H3_OUTPUT_ZERO_POINT,
        CUTE_YOLO_H4_DW_OUTPUT_SCALE,
        CUTE_YOLO_H4_DW_OUTPUT_ZERO_POINT,
        ACT_RELU);

    h4dw.depth_multiplier = 1;

    make_conv(
        h4pw, 1, 0, 1,
        CUTE_YOLO_HYBRID_EFFICIENT_OUT,
        w_h4pw, b_h4pw, m_h4pw, s_h4pw,
        CUTE_YOLO_H4_DW_OUTPUT_SCALE,
        CUTE_YOLO_H4_DW_OUTPUT_ZERO_POINT,
        CUTE_YOLO_H4_OUTPUT_SCALE,
        CUTE_YOLO_H4_OUTPUT_ZERO_POINT,
        ACT_RELU);

    // ----------------------------------------------------------
    // Hybrid 5
    // ----------------------------------------------------------

    make_conv(
        h5a, 3, 1, 1,
        CUTE_YOLO_HYBRID_NORMAL_OUT,
        w_h5a, b_h5a, m_h5a, s_h5a,
        CUTE_YOLO_H4_OUTPUT_SCALE,
        CUTE_YOLO_H4_OUTPUT_ZERO_POINT,
        CUTE_YOLO_H5_OUTPUT_SCALE,
        CUTE_YOLO_H5_OUTPUT_ZERO_POINT,
        ACT_RELU);

    make_conv(
        h5dw, 3, 1, 1,
        32,
        w_h5dw, b_h5dw, m_h5dw, s_h5dw,
        CUTE_YOLO_H4_OUTPUT_SCALE,
        CUTE_YOLO_H4_OUTPUT_ZERO_POINT,
        CUTE_YOLO_H5_DW_OUTPUT_SCALE,
        CUTE_YOLO_H5_DW_OUTPUT_ZERO_POINT,
        ACT_RELU);

    h5dw.depth_multiplier = 1;

    make_conv(
        h5pw, 1, 0, 1,
        CUTE_YOLO_HYBRID_EFFICIENT_OUT,
        w_h5pw, b_h5pw, m_h5pw, s_h5pw,
        CUTE_YOLO_H5_DW_OUTPUT_SCALE,
        CUTE_YOLO_H5_DW_OUTPUT_ZERO_POINT,
        CUTE_YOLO_H5_OUTPUT_SCALE,
        CUTE_YOLO_H5_OUTPUT_ZERO_POINT,
        ACT_RELU);

    // ----------------------------------------------------------
    // Linear detector head: 32x16x16 -> 5x16x16
    // ----------------------------------------------------------

    make_conv(
        head, 1, 0, 1, 5,
        w_head, b_head, m_head, s_head,
        CUTE_YOLO_H5_OUTPUT_SCALE,
        CUTE_YOLO_H5_OUTPUT_ZERO_POINT,
        CUTE_YOLO_OUTPUT_SCALE,
        CUTE_YOLO_OUTPUT_ZERO_POINT,
        ACT_NONE);

    // ----------------------------------------------------------
    // Execution:
    //
    // S1 X -> A
    // S2 A -> B
    // S3 B -> A
    // H1 A -> B
    // H2 B -> A
    // H3 A -> B
    // H4 B -> A
    // H5 A -> B
    // Head B -> A
    // ----------------------------------------------------------

    if (!run_split_conv_layer(
            "S1", &X, &A,
            s1a, s1b))
    {
        return false;
    }

    if (!run_split_conv_layer(
            "S2", &A, &B,
            s2a, s2b))
    {
        return false;
    }

    if (!run_split_conv_layer(
            "S3", &B, &A,
            s3a, s3b))
    {
        return false;
    }

    if (!run_hybrid_layer(
            "H1", &A, &B,
            h1a, h1dw, h1pw))
    {
        return false;
    }

    if (!run_hybrid_layer(
            "H2", &B, &A,
            h2a, h2dw, h2pw))
    {
        return false;
    }

    if (!run_hybrid_layer(
            "H3", &A, &B,
            h3a, h3dw, h3pw))
    {
        return false;
    }

    if (!run_hybrid_layer(
            "H4", &B, &A,
            h4a, h4dw, h4pw))
    {
        return false;
    }

    if (!run_hybrid_layer(
            "H5", &A, &B,
            h5a, h5dw, h5pw))
    {
        return false;
    }

    const uint32_t head_t0 =
        micros();

    const uint16_t head_w =
        noodle_conv2d(
            &B,
            &A,
            head,
            p);

#if V8_PRINT_LAYER_TIMING
    const uint32_t head_us =
        micros() - head_t0;

    Serial.printf(
        "HEAD  %luus -> %ux%ux%u\n",
        (unsigned long)head_us,
        (unsigned)A.C,
        (unsigned)A.W,
        (unsigned)A.W);
#endif

    if (!head_w)
    {
        return false;
    }

    return decode(
        &A,
        result);
}

// ============================================================
// Draw detections
// ============================================================

static void draw_box(
    const Detection &d,
    uint16_t color)
{

    int x1 =
        GUIDE_X + (int)(d.x1 * GUIDE_W);
    int y1 =
        GUIDE_Y + (int)(d.y1 * GUIDE_H);
    int x2 =
        GUIDE_X + (int)(d.x2 * GUIDE_W);
    int y2 =
        GUIDE_Y + (int)(d.y2 * GUIDE_H);

    x1 = constrain(
        x1, GUIDE_X, GUIDE_X + GUIDE_W - 1);
    y1 = constrain(
        y1, GUIDE_Y, GUIDE_Y + GUIDE_H - 1);
    x2 = constrain(
        x2, GUIDE_X, GUIDE_X + GUIDE_W - 1);
    y2 = constrain(
        y2, GUIDE_Y, GUIDE_Y + GUIDE_H - 1);

    const int box_w =
        x2 - x1 + 1;
    const int box_h =
        y2 - y1 + 1;

    tft.drawRect(
        x1, y1,
        box_w,
        box_h,
        color);

    if (tft_canvas)
    {
        tft_canvas->drawRect(
            x1, y1,
            box_w,
            box_h,
            color);
    }

#if CUTE_TFT_DRAW_CONFIDENCE
    char caption[8];

    const int pct =
        constrain(
            (int)lroundf(d.confidence * 100.0f),
            0, 100);

    snprintf(
        caption,
        sizeof(caption),
        "%d%%",
        pct);

    const int text_w =
        (int)strlen(caption) * 6;
    const int text_h = 8;

    int tx = x1;
    int ty = y1 - text_h;

    if (ty < GUIDE_Y)
    {
        ty = min(
            y1 + 1,
            GUIDE_Y + GUIDE_H - text_h);
    }

    tx = constrain(
        tx,
        0,
        TFT_FRAME_W - text_w);

    ty = constrain(
        ty,
        0,
        PREVIEW_H - text_h);

    tft.fillRect(
        tx, ty,
        text_w, text_h,
        ST77XX_BLACK);

    tft.setCursor(tx, ty);
    tft.setTextSize(1);
    tft.setTextColor(color, ST77XX_BLACK);
    tft.print(caption);

    if (tft_canvas)
    {
        tft_canvas->fillRect(
            tx, ty,
            text_w, text_h,
            ST77XX_BLACK);

        tft_canvas->setCursor(tx, ty);
        tft_canvas->setTextSize(1);
        tft_canvas->setTextColor(color, ST77XX_BLACK);
        tft_canvas->print(caption);
    }
#endif
}

static void draw_detections(
    const DetectionSet &set,
    uint32_t inference_us)
{

    static const uint16_t colors[] = {
        ST77XX_GREEN,
        ST77XX_CYAN,
        ST77XX_YELLOW,
        ST77XX_MAGENTA,
        ST77XX_WHITE,
    };

    for (uint8_t i = 0; i < set.count; ++i)
    {
        draw_box(
            set.items[i],
            colors[i % 5]);
    }

    // Bottom status line: number of detections + inference time.
    // Use integer milliseconds to keep the display compact and avoid
    // unnecessary floating-point formatting.
    const uint32_t inference_ms =
        (inference_us + 500UL) / 1000UL;

    char text[24];
    snprintf(
        text, sizeof(text),
        "N=%u T=%lums",
        (unsigned)set.count,
        (unsigned long)inference_ms);

    status(text);
}

// ============================================================
// Button helpers
// ============================================================

static bool button_pressed()
{
    if (digitalRead(BUTTON_PIN) != LOW)
    {
        return false;
    }

    delay(30);

    if (digitalRead(BUTTON_PIN) != LOW)
    {
        return false;
    }

    while (digitalRead(BUTTON_PIN) == LOW)
    {
        delay(10);
    }

    return true;
}

static void wait_for_button()
{
    while (digitalRead(BUTTON_PIN) == HIGH)
    {
        poll_serial_dataset_ready();
        delay(20);
    }

    delay(30);

    while (digitalRead(BUTTON_PIN) == LOW)
    {
        poll_serial_dataset_ready();
        delay(10);
    }
}

// ============================================================
// Setup
// ============================================================

void setup()
{
    // Built-in BOOT button = GPIO0. It remains the ROM boot strap only while
    // resetting; after boot it is used as the detector trigger.
    pinMode(
        BUTTON_PIN,
        INPUT_PULLUP);

    NoodleSerial::begin(
        921600);

    NoodleSerial::clear_input();

    delay(100);

    // Keep the on-board addressable RGB LED dark.
    rgbLedWrite(48, 0, 0, 0);
    pinMode(48, OUTPUT);
    digitalWrite(48, LOW);

    // Fixed firmware: load the current model from a raw flash partition, then
    // expose BLE provisioning for future .cute model updates.
    cute_model_begin();
    cute_model_ble_begin("Cute-YOLO");

    init_tft();

    if (!init_tft_shadow())
    {
        status(
            "TFT BUF FAIL",
            ST77XX_RED);

        Serial.println(
            "Could not allocate 160x128 RGB565 TFT shadow buffer.");

        while (true)
        {
            delay(1000);
        }
    }

    Serial.printf(
        "TFT shadow buffer: %ux%u RGB565 = %u bytes\n",
        (unsigned)TFT_FRAME_W,
        (unsigned)TFT_FRAME_H,
        (unsigned)TFT_FRAME_BYTES);

    if (!init_sample_gray())
    {
        status(
            "DATA MEM FAIL",
            ST77XX_RED);

        Serial.println(
            "Could not allocate 128x128 GRAY8 dataset buffer.");

        while (true)
        {
            delay(1000);
        }
    }

    Serial.printf(
        "Dataset buffer: %ux%u GRAY8 = %u bytes\n",
        (unsigned)IMG_W,
        (unsigned)IMG_H,
        (unsigned)SAMPLE_BYTES);

    if (!init_camera())
    {
        status(
            "CAM FAIL",
            ST77XX_RED);

        while (true)
        {
            delay(1000);
        }
    }

    noodle_tensor_init(&X);
    noodle_tensor_init(&A);
    noodle_tensor_init(&B);
    noodle_tensor_init(&BR0);
    noodle_tensor_init(&BR1);
    noodle_tensor_init(&DWTMP);

    // ----------------------------------------------------------
    // Grow every tensor to its worst-case retained capacity BEFORE
    // worker tasks start. No arena growth is allowed during parallel
    // inference.
    // ----------------------------------------------------------

    noodle_tensor_set_quantization(
        &X,
        CUTE_YOLO_INPUT_SCALE,
        CUTE_YOLO_INPUT_ZERO_POINT);

    bool memory_ok = true;

    memory_ok &=
        noodle_tensor_require_2d(
            &X,
            1,
            128) != nullptr;

    // A: largest full tensor = Stem1 output 8x64x64.
    memory_ok &=
        noodle_tensor_require_2d(
            &A,
            8,
            64) != nullptr;

    // B: largest full tensor = Stem2 output 16x32x32.
    memory_ok &=
        noodle_tensor_require_2d(
            &B,
            16,
            32) != nullptr;

    // BR0: largest branch = Stem1 A, 4x64x64.
    memory_ok &=
        noodle_tensor_require_2d(
            &BR0,
            4,
            64) != nullptr;

    // BR1: largest branch = Stem1 B, 4x64x64.
    // Hybrid 24x16x16 is smaller.
    memory_ok &=
        noodle_tensor_require_2d(
            &BR1,
            4,
            64) != nullptr;

    // Depthwise intermediate for every hybrid block.
    memory_ok &=
        noodle_tensor_require_2d(
            &DWTMP,
            32,
            16) != nullptr;

    if (!memory_ok)
    {
        status(
            "MEM FAIL",
            ST77XX_RED);

        Serial.println(
            "V8 tensor preallocation failed.");

        while (true)
        {
            delay(1000);
        }
    }

    Serial.printf(
        "V8 tensor capacities: "
        "X=%u A=%u B=%u BR0=%u BR1=%u DWTMP=%u bytes\n",
        (unsigned)noodle_tensor_capacity_bytes(&X),
        (unsigned)noodle_tensor_capacity_bytes(&A),
        (unsigned)noodle_tensor_capacity_bytes(&B),
        (unsigned)noodle_tensor_capacity_bytes(&BR0),
        (unsigned)noodle_tensor_capacity_bytes(&BR1),
        (unsigned)noodle_tensor_capacity_bytes(&DWTMP));

    const size_t retained_bytes =
        noodle_tensor_capacity_bytes(&X) +
        noodle_tensor_capacity_bytes(&A) +
        noodle_tensor_capacity_bytes(&B) +
        noodle_tensor_capacity_bytes(&BR0) +
        noodle_tensor_capacity_bytes(&BR1) +
        noodle_tensor_capacity_bytes(&DWTMP);

    Serial.printf(
        "V8 retained Noodle tensor bytes: %u\n",
        (unsigned)retained_bytes);

    if (!init_dualcore_workers())
    {
        status(
            "CORE FAIL",
            ST77XX_RED);

        Serial.println(
            "Could not start dual-core V8 workers.");

        while (true)
        {
            delay(1000);
        }
    }

    Serial.println(
        "Cute-YOLO fixed 8+24 dual-core firmware ready.");

    if (cute_model_ready())
    {
        Serial.printf(
            "Active model: %s  conf=%.2f nms=%.2f\n",
            cute_model_label(),
            (double)confidence_threshold(),
            (double)nms_iou_threshold());
        status("CUTE READY");
    }
    else
    {
        Serial.println(
            "No valid .cute model found. Upload one over BLE.");
        status("BLE: UPLOAD", ST77XX_YELLOW);
    }

    Serial.println(
        "SERIALDATA READY -- dataset collector should send RDYSAMPLE.");
    Serial.println(
        "DISPLAYDATA READY -- exact TFT collector should send RDYDISPLAY.");
    NoodleSerial::print_ready();
}

// ============================================================
// Main loop
// ============================================================

void loop()
{
    // --------------------------------------------------------
    // BLE MODEL MANAGEMENT FIRST
    // Do not touch camera/TFT/shadow buffer while flash upload
    // is active.
    // --------------------------------------------------------

    poll_serial_dataset_ready();

    cute_model_poll();

    static bool upload_ui_shown = false;

    if (cute_model_uploading())
    {
        if (!upload_ui_shown)
        {
            status(
                "BLE UPDATE",
                ST77XX_YELLOW);

            upload_ui_shown = true;
        }

        delay(20);
        return;
    }

    upload_ui_shown = false;

    if (!cute_model_ready())
    {
        status(
            "BLE: UPLOAD",
            ST77XX_YELLOW);

        delay(20);
        return;
    }

    // --------------------------------------------------------
    // Only touch the camera after the model manager is idle.
    // --------------------------------------------------------

    camera_fb_t *fb =
        esp_camera_fb_get();

    if (!fb)
    {
        return;
    }

    draw_preview(fb);

    if (!button_pressed())
    {
        esp_camera_fb_return(fb);
        return;
    }

    // The TFT keeps the current color preview frozen.
    status("RUNNING", ST77XX_YELLOW);

    const bool input_ok =
        frame_to_tensor(fb);

    esp_camera_fb_return(fb);

    if (!input_ok)
    {
        status("INPUT ERR", ST77XX_RED);
        delay(1000);
        return;
    }

    DetectionSet detections = {};

    // Time only the actual Cute-YOLO inference path:
    // Noodle convolutions + output decode/NMS.
    // Camera capture, TFT preview, and RGB565->128x128 preprocessing
    // are intentionally excluded.
    const uint32_t inference_t0 = micros();

    cute_model_set_inference_busy(true);

    const bool prediction_ok =
        predict(&detections);

    cute_model_set_inference_busy(false);

    const uint32_t inference_us =
        micros() - inference_t0;

    if (!prediction_ok)
    {
        status("PRED ERR", ST77XX_RED);
        delay(1000);
        return;
    }

    draw_detections(
        detections,
        inference_us);

    // Send the clean 128x128 detector image plus predicted box metadata
    // only when a live PC dataset collector has advertised readiness.
    // The Python side draws the labeled preview, preserving a pristine
    // raw training image on disk.
    send_dataset_sample_if_ready(
        detections,
        inference_us);

    // Exact rendered-TFT capture is independent of the clean training-data
    // collector and runs only when a PC sends RDYDISPLAY.
    send_display_frame_if_ready(
        detections,
        inference_us);

    // Keep detections on screen until the next button press.
    wait_for_button();

    tft.fillScreen(ST77XX_BLACK);

    if (tft_canvas)
    {
        tft_canvas->fillScreen(ST77XX_BLACK);
    }
}