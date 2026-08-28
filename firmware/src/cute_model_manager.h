#pragma once

#include <Arduino.h>
#include <stdint.h>
#include <stddef.h>
#include "esp_partition.h"

static constexpr uint16_t CUTE_MODEL_FORMAT_VERSION = 1;
static constexpr uint16_t CUTE_MODEL_HEADER_BYTES = 1024;
static constexpr uint16_t CUTE_MODEL_LAYER_COUNT = 22;
static constexpr uint32_t CUTE_MODEL_ARCH_ID = 0xE88F28A8u;

struct __attribute__((packed)) CuteLayerRecord {
  uint32_t weight_offset;
  uint32_t weight_count;
  uint32_t bias_offset;
  uint32_t bias_count;
  uint32_t multiplier_offset;
  uint32_t shift_offset;
  float input_scale;
  float output_scale;
  int32_t input_zero_point;
  int32_t output_zero_point;
};

struct __attribute__((packed)) CuteModelHeader {
  char magic[4];
  uint16_t version;
  uint16_t header_bytes;
  uint32_t architecture_id;
  uint32_t total_bytes;
  uint32_t payload_crc32;
  uint16_t layer_count;
  uint16_t flags;
  char label[24];
  float confidence_threshold;
  float nms_iou_threshold;
  float min_box_w;
  float min_box_h;
  uint8_t max_detections;
  uint8_t reserved0[3];
  float input_scale;
  int32_t input_zero_point;
  float output_scale;
  int32_t output_zero_point;
  char architecture_name[40];
  uint32_t header_crc32;
  CuteLayerRecord layers[CUTE_MODEL_LAYER_COUNT];
  uint8_t reserved2[16];
};

static_assert(sizeof(CuteLayerRecord) == 40, "CuteLayerRecord must be 40 bytes");
static_assert(sizeof(CuteModelHeader) == CUTE_MODEL_HEADER_BYTES,
              "CuteModelHeader must be exactly 1024 bytes");

enum CuteLayerId : uint8_t {
  CUTE_LAYER_S1A = 0,
  CUTE_LAYER_S1B,
  CUTE_LAYER_S2A,
  CUTE_LAYER_S2B,
  CUTE_LAYER_S3A,
  CUTE_LAYER_S3B,
  CUTE_LAYER_H1A,
  CUTE_LAYER_H1DW,
  CUTE_LAYER_H1PW,
  CUTE_LAYER_H2A,
  CUTE_LAYER_H2DW,
  CUTE_LAYER_H2PW,
  CUTE_LAYER_H3A,
  CUTE_LAYER_H3DW,
  CUTE_LAYER_H3PW,
  CUTE_LAYER_H4A,
  CUTE_LAYER_H4DW,
  CUTE_LAYER_H4PW,
  CUTE_LAYER_H5A,
  CUTE_LAYER_H5DW,
  CUTE_LAYER_H5PW,
  CUTE_LAYER_HEAD,
};

bool cute_model_begin();
bool cute_model_ble_begin(const char *device_name = "Cute-YOLO");
void cute_model_poll();

bool cute_model_ready();
bool cute_model_uploading();
void cute_model_set_inference_busy(bool busy);

const CuteModelHeader *cute_model_header();
const CuteLayerRecord *cute_model_layer(CuteLayerId id);
const uint8_t *cute_model_base();

const int8_t *cute_model_weight(CuteLayerId id);
const int32_t *cute_model_bias(CuteLayerId id);
const int32_t *cute_model_multiplier(CuteLayerId id);
const int32_t *cute_model_shift(CuteLayerId id);

const char *cute_model_label();
float cute_model_confidence_threshold();
float cute_model_nms_iou_threshold();
float cute_model_min_box_w();
float cute_model_min_box_h();
uint8_t cute_model_max_detections();
const char *cute_model_status();
