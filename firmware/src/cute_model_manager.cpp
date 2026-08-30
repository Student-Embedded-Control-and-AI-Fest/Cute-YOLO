#include "cute_model_manager.h"

#include <Arduino.h>
#include <Preferences.h>
#include <NimBLEDevice.h>
#include <string.h>
#include <stdio.h>
#include <stdarg.h>
#include <string>

namespace
{

  static constexpr const char *PART_A_LABEL = "cuteA";
  static constexpr const char *PART_B_LABEL = "cuteB";

  // Must match the current fixed firmware/runtime capacity.
  // Older .cute packages with smaller values remain valid.
  static constexpr uint8_t CUTE_MODEL_RUNTIME_MAX_DETECTIONS = 32;

  static constexpr const char *SERVICE_UUID =
      "7f1d0001-9f3b-4c2b-8f3e-5b51c0de0001";
  static constexpr const char *CTRL_UUID =
      "7f1d0002-9f3b-4c2b-8f3e-5b51c0de0001";
  static constexpr const char *DATA_UUID =
      "7f1d0003-9f3b-4c2b-8f3e-5b51c0de0001";
  static constexpr const char *STATUS_UUID =
      "7f1d0004-9f3b-4c2b-8f3e-5b51c0de0001";

  static const esp_partition_t *g_part_a = nullptr;
  static const esp_partition_t *g_part_b = nullptr;
  static const esp_partition_t *g_active_part = nullptr;

  static const uint8_t *g_model_base = nullptr;
  static esp_partition_mmap_handle_t g_mmap_handle = 0;
  static bool g_mapped = false;
  static char g_active_slot = '\0';

  static volatile bool g_uploading = false;
  static volatile bool g_inference_busy = false;
  static const esp_partition_t *g_upload_part = nullptr;
  static char g_upload_slot = '\0';
  static size_t g_upload_expected = 0;
  static size_t g_upload_received = 0;
  static volatile char g_pending_slot = '\0';

  static NimBLECharacteristic *g_status_characteristic = nullptr;
  static char g_status[96] = "BOOT";
  static Preferences g_preferences;

  static const uint32_t EXPECTED_WEIGHT_COUNTS[CUTE_MODEL_LAYER_COUNT] = {
      36, 36, 576, 576, 2304, 2304,
      2304, 288, 768,
      2304, 288, 768,
      2304, 288, 768,
      2304, 288, 768,
      2304, 288, 768,
      160};

  static const uint32_t EXPECTED_BIAS_COUNTS[CUTE_MODEL_LAYER_COUNT] = {
      4, 4, 8, 8, 16, 16,
      8, 32, 24,
      8, 32, 24,
      8, 32, 24,
      8, 32, 24,
      8, 32, 24,
      5};

  static uint32_t crc32_update(uint32_t crc, const uint8_t *data, size_t n)
  {
    while (n--)
    {
      crc ^= *data++;
      for (uint8_t k = 0; k < 8; ++k)
      {
        const uint32_t mask = (uint32_t)-(int32_t)(crc & 1u);
        crc = (crc >> 1) ^ (0xEDB88320u & mask);
      }
    }
    return crc;
  }

  static uint32_t crc32_memory(const uint8_t *data, size_t n)
  {
    uint32_t crc = 0xFFFFFFFFu;
    crc = crc32_update(crc, data, n);
    return crc ^ 0xFFFFFFFFu;
  }

  static bool crc32_partition(const esp_partition_t *part,
                              size_t offset,
                              size_t n,
                              uint32_t *out_crc)
  {
    if (!part || !out_crc)
      return false;
    uint8_t buffer[512];
    uint32_t crc = 0xFFFFFFFFu;
    while (n > 0)
    {
      const size_t chunk = n < sizeof(buffer) ? n : sizeof(buffer);
      if (esp_partition_read(part, offset, buffer, chunk) != ESP_OK)
      {
        return false;
      }
      crc = crc32_update(crc, buffer, chunk);
      offset += chunk;
      n -= chunk;
    }
    *out_crc = crc ^ 0xFFFFFFFFu;
    return true;
  }

  static bool range_ok(uint32_t offset,
                       uint32_t bytes,
                       uint32_t total_bytes)
  {
    if (offset < CUTE_MODEL_HEADER_BYTES)
      return false;
    if (offset > total_bytes)
      return false;
    if (bytes > total_bytes - offset)
      return false;
    return true;
  }

  static bool validation_fail(char *reason,
                              size_t reason_size,
                              const char *fmt,
                              ...)
  {
    if (reason && reason_size > 0)
    {
      va_list args;
      va_start(args, fmt);
      vsnprintf(reason, reason_size, fmt, args);
      va_end(args);
    }
    return false;
  }

  static bool validate_model_partition(const esp_partition_t *part,
                                       CuteModelHeader *out_header = nullptr,
                                       char *reason = nullptr,
                                       size_t reason_size = 0)
  {
    if (!part)
      return validation_fail(reason, reason_size, "partition=null");

    if (part->size < CUTE_MODEL_HEADER_BYTES)
      return validation_fail(reason, reason_size,
                             "partition too small: %u",
                             (unsigned)part->size);

    CuteModelHeader header = {};
    const esp_err_t read_header_err =
        esp_partition_read(part, 0, &header, sizeof(header));

    if (read_header_err != ESP_OK)
      return validation_fail(reason, reason_size,
                             "header read err=%d",
                             (int)read_header_err);

    if (memcmp(header.magic, "CUTE", 4) != 0)
      return validation_fail(reason, reason_size,
                             "magic %.4s",
                             header.magic);

    if (header.version != CUTE_MODEL_FORMAT_VERSION)
      return validation_fail(reason, reason_size,
                             "version %u != %u",
                             (unsigned)header.version,
                             (unsigned)CUTE_MODEL_FORMAT_VERSION);

    if (header.header_bytes != CUTE_MODEL_HEADER_BYTES)
      return validation_fail(reason, reason_size,
                             "header_bytes %u != %u",
                             (unsigned)header.header_bytes,
                             (unsigned)CUTE_MODEL_HEADER_BYTES);

    if (header.architecture_id != CUTE_MODEL_ARCH_ID)
      return validation_fail(reason, reason_size,
                             "arch %08lX != %08lX",
                             (unsigned long)header.architecture_id,
                             (unsigned long)CUTE_MODEL_ARCH_ID);

    if (header.layer_count != CUTE_MODEL_LAYER_COUNT)
      return validation_fail(reason, reason_size,
                             "layers %u != %u",
                             (unsigned)header.layer_count,
                             (unsigned)CUTE_MODEL_LAYER_COUNT);

    if (header.total_bytes < CUTE_MODEL_HEADER_BYTES ||
        header.total_bytes > part->size)
      return validation_fail(reason, reason_size,
                             "total_bytes %lu invalid",
                             (unsigned long)header.total_bytes);

    const uint32_t expected_header_crc = header.header_crc32;
    header.header_crc32 = 0;

    const uint32_t actual_header_crc =
        crc32_memory(
            reinterpret_cast<const uint8_t *>(&header),
            sizeof(header));

    if (actual_header_crc != expected_header_crc)
      return validation_fail(reason, reason_size,
                             "header CRC exp=%08lX got=%08lX",
                             (unsigned long)expected_header_crc,
                             (unsigned long)actual_header_crc);

    header.header_crc32 = expected_header_crc;

    if (header.max_detections == 0 ||
        header.max_detections > CUTE_MODEL_RUNTIME_MAX_DETECTIONS)
      return validation_fail(reason, reason_size,
                             "max_det %u > %u",
                             (unsigned)header.max_detections,
                             (unsigned)CUTE_MODEL_RUNTIME_MAX_DETECTIONS);

    if (!(header.confidence_threshold >= 0.0f &&
          header.confidence_threshold <= 1.0f))
      return validation_fail(reason, reason_size,
                             "confidence %.4f invalid",
                             (double)header.confidence_threshold);

    if (!(header.nms_iou_threshold >= 0.0f &&
          header.nms_iou_threshold <= 1.0f))
      return validation_fail(reason, reason_size,
                             "nms %.4f invalid",
                             (double)header.nms_iou_threshold);

    if (!(header.min_box_w >= 0.0f && header.min_box_w <= 1.0f))
      return validation_fail(reason, reason_size,
                             "min_box_w %.4f invalid",
                             (double)header.min_box_w);

    if (!(header.min_box_h >= 0.0f && header.min_box_h <= 1.0f))
      return validation_fail(reason, reason_size,
                             "min_box_h %.4f invalid",
                             (double)header.min_box_h);

    if (header.label[23] != '\0')
      return validation_fail(reason, reason_size,
                             "label not terminated");

    for (uint16_t i = 0; i < CUTE_MODEL_LAYER_COUNT; ++i)
    {
      const CuteLayerRecord &r = header.layers[i];

      if (r.weight_count != EXPECTED_WEIGHT_COUNTS[i])
        return validation_fail(reason, reason_size,
                               "L%u weight_count %lu != %lu",
                               (unsigned)i,
                               (unsigned long)r.weight_count,
                               (unsigned long)EXPECTED_WEIGHT_COUNTS[i]);

      if (r.bias_count != EXPECTED_BIAS_COUNTS[i])
        return validation_fail(reason, reason_size,
                               "L%u bias_count %lu != %lu",
                               (unsigned)i,
                               (unsigned long)r.bias_count,
                               (unsigned long)EXPECTED_BIAS_COUNTS[i]);

      if (!range_ok(r.weight_offset,
                    r.weight_count,
                    header.total_bytes))
        return validation_fail(reason, reason_size,
                               "L%u weight range",
                               (unsigned)i);

      if (!range_ok(r.bias_offset,
                    r.bias_count * sizeof(int32_t),
                    header.total_bytes))
        return validation_fail(reason, reason_size,
                               "L%u bias range",
                               (unsigned)i);

      if (!range_ok(r.multiplier_offset,
                    r.bias_count * sizeof(int32_t),
                    header.total_bytes))
        return validation_fail(reason, reason_size,
                               "L%u multiplier range",
                               (unsigned)i);

      if (!range_ok(r.shift_offset,
                    r.bias_count * sizeof(int32_t),
                    header.total_bytes))
        return validation_fail(reason, reason_size,
                               "L%u shift range",
                               (unsigned)i);

      if ((r.bias_offset & 3u) ||
          (r.multiplier_offset & 3u) ||
          (r.shift_offset & 3u))
        return validation_fail(reason, reason_size,
                               "L%u alignment",
                               (unsigned)i);

      if (!(r.input_scale > 0.0f))
        return validation_fail(reason, reason_size,
                               "L%u input_scale %.7g",
                               (unsigned)i,
                               (double)r.input_scale);

      if (!(r.output_scale > 0.0f))
        return validation_fail(reason, reason_size,
                               "L%u output_scale %.7g",
                               (unsigned)i,
                               (double)r.output_scale);
    }

    uint32_t payload_crc = 0;

    if (!crc32_partition(
            part,
            CUTE_MODEL_HEADER_BYTES,
            header.total_bytes - CUTE_MODEL_HEADER_BYTES,
            &payload_crc))
      return validation_fail(reason, reason_size,
                             "payload read failed");

    if (payload_crc != header.payload_crc32)
      return validation_fail(reason, reason_size,
                             "payload CRC exp=%08lX got=%08lX",
                             (unsigned long)header.payload_crc32,
                             (unsigned long)payload_crc);

    if (out_header)
      *out_header = header;

    if (reason && reason_size > 0)
      snprintf(reason, reason_size, "OK");

    return true;
  }

  static const esp_partition_t *partition_for_slot(char slot)
  {
    return slot == 'B' ? g_part_b : g_part_a;
  }

  static void set_status(const char *text)
  {
    snprintf(g_status, sizeof(g_status), "%s", text ? text : "");
    Serial.printf("[CuteBLE] %s\n", g_status);
    if (g_status_characteristic)
    {
      g_status_characteristic->setValue(
          reinterpret_cast<const uint8_t *>(g_status), strlen(g_status));
      g_status_characteristic->notify();
    }
  }

  static bool map_partition(const esp_partition_t *part, char slot)
  {
    CuteModelHeader check = {};
    if (!validate_model_partition(part, &check))
      return false;

    const void *new_ptr = nullptr;
    esp_partition_mmap_handle_t new_handle = 0;
    if (esp_partition_mmap(part,
                           0,
                           check.total_bytes,
                           ESP_PARTITION_MMAP_DATA,
                           &new_ptr,
                           &new_handle) != ESP_OK)
    {
      return false;
    }

    if (g_mapped)
    {
      esp_partition_munmap(g_mmap_handle);
    }

    g_model_base = reinterpret_cast<const uint8_t *>(new_ptr);
    g_mmap_handle = new_handle;
    g_mapped = true;
    g_active_part = part;
    g_active_slot = slot;
    return true;
  }

  static void handle_begin(size_t expected_bytes)
  {
    if (g_uploading)
    {
      set_status("ERR already uploading");
      return;
    }
    if (!g_part_a || !g_part_b)
    {
      set_status("ERR model partitions");
      return;
    }

    // Stop new inference immediately, then wait for any already-running
    // inference to finish before erasing/programming flash.
    g_uploading = true;

    const uint32_t wait_t0 = millis();
    while (g_inference_busy && (millis() - wait_t0) < 3000)
    {
      delay(5);
    }

    if (g_inference_busy)
    {
      g_uploading = false;
      set_status("ERR inference busy");
      return;
    }

    const char slot = g_active_slot == 'A' ? 'B' : 'A';
    const esp_partition_t *target = partition_for_slot(slot);
    if (!target || expected_bytes < CUTE_MODEL_HEADER_BYTES ||
        expected_bytes > target->size)
    {
      g_uploading = false;
      set_status("ERR model size");
      return;
    }

    const size_t erase_size = target->erase_size;
    const size_t erase_bytes =
        ((expected_bytes + erase_size - 1) / erase_size) * erase_size;

    set_status("ERASING");
    if (esp_partition_erase_range(target, 0, erase_bytes) != ESP_OK)
    {
      g_uploading = false;
      set_status("ERR erase");
      return;
    }

    g_upload_part = target;
    g_upload_slot = slot;
    g_upload_expected = expected_bytes;
    g_upload_received = 0;

    char msg[64];
    snprintf(msg, sizeof(msg), "READY %c %u", slot, (unsigned)expected_bytes);
    set_status(msg);
  }

  static void handle_data(const uint8_t *data, size_t n)
  {
    if (!g_uploading || !g_upload_part || !data || n == 0)
      return;

    if (g_upload_received + n > g_upload_expected)
    {
      g_uploading = false;
      set_status("ERR overflow");
      return;
    }

    if (esp_partition_write(g_upload_part,
                            g_upload_received,
                            data,
                            n) != ESP_OK)
    {
      g_uploading = false;
      set_status("ERR flash write");
      return;
    }

    g_upload_received += n;

    if ((g_upload_received & 0xFFFu) < n ||
        g_upload_received == g_upload_expected)
    {
      char msg[64];
      snprintf(msg, sizeof(msg), "RX %u/%u",
               (unsigned)g_upload_received,
               (unsigned)g_upload_expected);
      set_status(msg);
    }
  }

  static void handle_end()
  {
    if (!g_uploading)
    {
      set_status("ERR no upload");
      return;
    }

    g_uploading = false;
    if (g_upload_received != g_upload_expected)
    {
      set_status("ERR incomplete");
      return;
    }

    CuteModelHeader header = {};
    char validation_reason[80] = {};
    set_status("VERIFY");

    if (!validate_model_partition(
            g_upload_part,
            &header,
            validation_reason,
            sizeof(validation_reason)))
    {
      char msg[96];
      snprintf(
          msg,
          sizeof(msg),
          "ERR %s",
          validation_reason[0]
              ? validation_reason
              : "model validation");
      set_status(msg);
      return;
    }

    // Defer mmap switching to the Arduino loop task so the old model cannot be
    // unmapped while an inference worker is still executing.
    g_pending_slot = g_upload_slot;

    char msg[80];
    snprintf(msg, sizeof(msg), "VALID %c %s", g_upload_slot, header.label);
    set_status(msg);
  }

  static void handle_abort()
  {
    g_uploading = false;
    g_upload_part = nullptr;
    g_upload_slot = '\0';
    g_upload_expected = 0;
    g_upload_received = 0;
    set_status("ABORTED");
  }

  static void handle_info()
  {
    if (!cute_model_ready())
    {
      set_status("INFO no model");
      return;
    }

    char msg[88];
    snprintf(msg, sizeof(msg), "INFO %c %s %.2f %.2f",
             g_active_slot,
             cute_model_label(),
             (double)cute_model_confidence_threshold(),
             (double)cute_model_nms_iou_threshold());
    set_status(msg);
  }

  static void handle_control(const std::string &command)
  {
    if (command.rfind("BEGIN ", 0) == 0)
    {
      unsigned long n = 0;
      if (sscanf(command.c_str(), "BEGIN %lu", &n) == 1)
      {
        handle_begin((size_t)n);
      }
      else
      {
        set_status("ERR BEGIN syntax");
      }
      return;
    }
    if (command == "END")
    {
      handle_end();
      return;
    }
    if (command == "ABORT")
    {
      handle_abort();
      return;
    }
    if (command == "INFO")
    {
      handle_info();
      return;
    }
    set_status("ERR command");
  }

  class ControlCallbacks : public NimBLECharacteristicCallbacks
  {
    void onWrite(NimBLECharacteristic *characteristic,
                 NimBLEConnInfo &connInfo) override
    {
      (void)connInfo;
      const NimBLEAttValue &value = characteristic->getValue();
      std::string command(
          reinterpret_cast<const char *>(value.data()), value.size());
      handle_control(command);
    }
  };

  class DataCallbacks : public NimBLECharacteristicCallbacks
  {
    void onWrite(NimBLECharacteristic *characteristic,
                 NimBLEConnInfo &connInfo) override
    {
      (void)connInfo;
      const NimBLEAttValue &value = characteristic->getValue();
      handle_data(value.data(), value.size());
    }
  };

  static ControlCallbacks g_control_callbacks;
  static DataCallbacks g_data_callbacks;

} // namespace

bool cute_model_begin()
{
  g_part_a = esp_partition_find_first(
      ESP_PARTITION_TYPE_DATA, ESP_PARTITION_SUBTYPE_ANY, PART_A_LABEL);
  g_part_b = esp_partition_find_first(
      ESP_PARTITION_TYPE_DATA, ESP_PARTITION_SUBTYPE_ANY, PART_B_LABEL);

  if (!g_part_a || !g_part_b)
  {
    set_status("ERR cuteA/cuteB partitions");
    return false;
  }

  g_preferences.begin("cute-model", false);
  const char preferred = (char)g_preferences.getUChar("slot", 'A');
  const char other = preferred == 'A' ? 'B' : 'A';

  if (map_partition(partition_for_slot(preferred), preferred))
  {
    char msg[80];
    snprintf(msg, sizeof(msg), "MODEL %c %s", preferred, cute_model_label());
    set_status(msg);
    return true;
  }

  if (map_partition(partition_for_slot(other), other))
  {
    g_preferences.putUChar("slot", (uint8_t)other);
    char msg[80];
    snprintf(msg, sizeof(msg), "MODEL %c %s", other, cute_model_label());
    set_status(msg);
    return true;
  }

  set_status("NO MODEL");
  return false;
}

bool cute_model_ble_begin(const char *device_name)
{
  NimBLEDevice::init(device_name ? device_name : "Cute-YOLO");
  NimBLEDevice::setMTU(247);

  NimBLEServer *server = NimBLEDevice::createServer();
  if (!server)
  {
    set_status("ERR BLE server");
    return false;
  }
  server->advertiseOnDisconnect(true);

  NimBLEService *service = server->createService(SERVICE_UUID);
  if (!service)
  {
    set_status("ERR BLE service");
    return false;
  }

  NimBLECharacteristic *control = service->createCharacteristic(
      CTRL_UUID,
      NIMBLE_PROPERTY::WRITE | NIMBLE_PROPERTY::READ,
      96);

  NimBLECharacteristic *data = service->createCharacteristic(
      DATA_UUID,
      NIMBLE_PROPERTY::WRITE | NIMBLE_PROPERTY::WRITE_NR,
      512);

  g_status_characteristic = service->createCharacteristic(
      STATUS_UUID,
      NIMBLE_PROPERTY::READ | NIMBLE_PROPERTY::NOTIFY,
      96);

  if (!control || !data || !g_status_characteristic)
  {
    set_status("ERR BLE characteristic");
    return false;
  }

  control->setCallbacks(&g_control_callbacks);
  data->setCallbacks(&g_data_callbacks);
  g_status_characteristic->setValue(
      reinterpret_cast<const uint8_t *>(g_status), strlen(g_status));

  service->start();

  NimBLEAdvertising *advertising = NimBLEDevice::getAdvertising();
  advertising->setName(device_name ? device_name : "Cute-YOLO");
  advertising->addServiceUUID(SERVICE_UUID);
  advertising->enableScanResponse(true);

  if (!advertising->start())
  {
    set_status("ERR BLE advertising");
    return false;
  }

  Serial.println("[CuteBLE] advertising as Cute-YOLO");
  return true;
}

void cute_model_poll()
{
  const char pending = g_pending_slot;
  if (pending == '\0')
    return;

  g_pending_slot = '\0';
  const esp_partition_t *part = partition_for_slot(pending);
  if (!map_partition(part, pending))
  {
    set_status("ERR activate");
    return;
  }

  g_preferences.putUChar("slot", (uint8_t)pending);
  g_upload_part = nullptr;
  g_upload_slot = '\0';
  g_upload_expected = 0;
  g_upload_received = 0;

  char msg[88];
  snprintf(msg, sizeof(msg), "ACTIVE %c %s", pending, cute_model_label());
  set_status(msg);
}

bool cute_model_ready()
{
  return g_mapped && g_model_base != nullptr;
}

bool cute_model_uploading()
{
  return g_uploading;
}

void cute_model_set_inference_busy(bool busy)
{
  g_inference_busy = busy;
}

const CuteModelHeader *cute_model_header()
{
  if (!cute_model_ready())
    return nullptr;
  return reinterpret_cast<const CuteModelHeader *>(g_model_base);
}

const CuteLayerRecord *cute_model_layer(CuteLayerId id)
{
  const CuteModelHeader *header = cute_model_header();
  if (!header || (uint8_t)id >= CUTE_MODEL_LAYER_COUNT)
    return nullptr;
  return &header->layers[(uint8_t)id];
}

const uint8_t *cute_model_base()
{
  return g_model_base;
}

const int8_t *cute_model_weight(CuteLayerId id)
{
  const CuteLayerRecord *r = cute_model_layer(id);
  return r ? reinterpret_cast<const int8_t *>(g_model_base + r->weight_offset) : nullptr;
}

const int32_t *cute_model_bias(CuteLayerId id)
{
  const CuteLayerRecord *r = cute_model_layer(id);
  return r ? reinterpret_cast<const int32_t *>(g_model_base + r->bias_offset) : nullptr;
}

const int32_t *cute_model_multiplier(CuteLayerId id)
{
  const CuteLayerRecord *r = cute_model_layer(id);
  return r ? reinterpret_cast<const int32_t *>(g_model_base + r->multiplier_offset) : nullptr;
}

const int32_t *cute_model_shift(CuteLayerId id)
{
  const CuteLayerRecord *r = cute_model_layer(id);
  return r ? reinterpret_cast<const int32_t *>(g_model_base + r->shift_offset) : nullptr;
}

const char *cute_model_label()
{
  const CuteModelHeader *header = cute_model_header();
  return header ? header->label : "none";
}

float cute_model_confidence_threshold()
{
  const CuteModelHeader *header = cute_model_header();
  return header ? header->confidence_threshold : 0.50f;
}

float cute_model_nms_iou_threshold()
{
  const CuteModelHeader *header = cute_model_header();
  return header ? header->nms_iou_threshold : 0.35f;
}

float cute_model_min_box_w()
{
  const CuteModelHeader *header = cute_model_header();
  return header ? header->min_box_w : 0.05f;
}

float cute_model_min_box_h()
{
  const CuteModelHeader *header = cute_model_header();
  return header ? header->min_box_h : 0.05f;
}

uint8_t cute_model_max_detections()
{
  const CuteModelHeader *header = cute_model_header();
  return header
             ? header->max_detections
             : CUTE_MODEL_RUNTIME_MAX_DETECTIONS;
}

const char *cute_model_status()
{
  return g_status;
}
