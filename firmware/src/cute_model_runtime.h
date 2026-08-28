#pragma once

#include "cute_model_manager.h"

#define CUTE_YOLO_FIXED_8_24 1
#define CUTE_YOLO_V8_REBALANCED_CNN_DW_PW 1
#define CUTE_YOLO_FIXED_HYBRID_8_24 1

#define CUTE_YOLO_INPUT_W 128
#define CUTE_YOLO_INPUT_H 128
#define CUTE_YOLO_INPUT_C 1
#define CUTE_YOLO_GRID_W 16
#define CUTE_YOLO_OUTPUT_C 5

#define CUTE_YOLO_STEM1_BRANCH_A_OUT 4
#define CUTE_YOLO_STEM1_BRANCH_B_OUT 4
#define CUTE_YOLO_STEM2_BRANCH_A_OUT 8
#define CUTE_YOLO_STEM2_BRANCH_B_OUT 8
#define CUTE_YOLO_STEM3_BRANCH_A_OUT 16
#define CUTE_YOLO_STEM3_BRANCH_B_OUT 16

#define CUTE_YOLO_HYBRID_NORMAL_OUT 8
#define CUTE_YOLO_HYBRID_EFFICIENT_OUT 24
#define CUTE_YOLO_HYBRID_CHANNELS 32
#define CUTE_YOLO_HYBRID_BLOCKS 5

#define CUTE_YOLO_INPUT_SCALE              (cute_model_ready() ? cute_model_header()->input_scale : (1.0f / 255.0f))
#define CUTE_YOLO_INPUT_ZERO_POINT         (cute_model_ready() ? cute_model_header()->input_zero_point : -128)
#define CUTE_YOLO_OUTPUT_SCALE             (cute_model_ready() ? cute_model_header()->output_scale : 1.0f)
#define CUTE_YOLO_OUTPUT_ZERO_POINT        (cute_model_ready() ? cute_model_header()->output_zero_point : 0)

#define CUTE_YOLO_STEM1_OUTPUT_SCALE       (cute_model_layer(CUTE_LAYER_S1A)->output_scale)
#define CUTE_YOLO_STEM1_OUTPUT_ZERO_POINT  (cute_model_layer(CUTE_LAYER_S1A)->output_zero_point)
#define CUTE_YOLO_STEM2_OUTPUT_SCALE       (cute_model_layer(CUTE_LAYER_S2A)->output_scale)
#define CUTE_YOLO_STEM2_OUTPUT_ZERO_POINT  (cute_model_layer(CUTE_LAYER_S2A)->output_zero_point)
#define CUTE_YOLO_STEM3_OUTPUT_SCALE       (cute_model_layer(CUTE_LAYER_S3A)->output_scale)
#define CUTE_YOLO_STEM3_OUTPUT_ZERO_POINT  (cute_model_layer(CUTE_LAYER_S3A)->output_zero_point)

#define CUTE_YOLO_H1_DW_OUTPUT_SCALE       (cute_model_layer(CUTE_LAYER_H1DW)->output_scale)
#define CUTE_YOLO_H1_DW_OUTPUT_ZERO_POINT  (cute_model_layer(CUTE_LAYER_H1DW)->output_zero_point)
#define CUTE_YOLO_H1_OUTPUT_SCALE          (cute_model_layer(CUTE_LAYER_H1A)->output_scale)
#define CUTE_YOLO_H1_OUTPUT_ZERO_POINT     (cute_model_layer(CUTE_LAYER_H1A)->output_zero_point)
#define CUTE_YOLO_H2_DW_OUTPUT_SCALE       (cute_model_layer(CUTE_LAYER_H2DW)->output_scale)
#define CUTE_YOLO_H2_DW_OUTPUT_ZERO_POINT  (cute_model_layer(CUTE_LAYER_H2DW)->output_zero_point)
#define CUTE_YOLO_H2_OUTPUT_SCALE          (cute_model_layer(CUTE_LAYER_H2A)->output_scale)
#define CUTE_YOLO_H2_OUTPUT_ZERO_POINT     (cute_model_layer(CUTE_LAYER_H2A)->output_zero_point)
#define CUTE_YOLO_H3_DW_OUTPUT_SCALE       (cute_model_layer(CUTE_LAYER_H3DW)->output_scale)
#define CUTE_YOLO_H3_DW_OUTPUT_ZERO_POINT  (cute_model_layer(CUTE_LAYER_H3DW)->output_zero_point)
#define CUTE_YOLO_H3_OUTPUT_SCALE          (cute_model_layer(CUTE_LAYER_H3A)->output_scale)
#define CUTE_YOLO_H3_OUTPUT_ZERO_POINT     (cute_model_layer(CUTE_LAYER_H3A)->output_zero_point)
#define CUTE_YOLO_H4_DW_OUTPUT_SCALE       (cute_model_layer(CUTE_LAYER_H4DW)->output_scale)
#define CUTE_YOLO_H4_DW_OUTPUT_ZERO_POINT  (cute_model_layer(CUTE_LAYER_H4DW)->output_zero_point)
#define CUTE_YOLO_H4_OUTPUT_SCALE          (cute_model_layer(CUTE_LAYER_H4A)->output_scale)
#define CUTE_YOLO_H4_OUTPUT_ZERO_POINT     (cute_model_layer(CUTE_LAYER_H4A)->output_zero_point)
#define CUTE_YOLO_H5_DW_OUTPUT_SCALE       (cute_model_layer(CUTE_LAYER_H5DW)->output_scale)
#define CUTE_YOLO_H5_DW_OUTPUT_ZERO_POINT  (cute_model_layer(CUTE_LAYER_H5DW)->output_zero_point)
#define CUTE_YOLO_H5_OUTPUT_SCALE          (cute_model_layer(CUTE_LAYER_H5A)->output_scale)
#define CUTE_YOLO_H5_OUTPUT_ZERO_POINT     (cute_model_layer(CUTE_LAYER_H5A)->output_zero_point)

#define w01a   (cute_model_weight(CUTE_LAYER_S1A))
#define b01a   (cute_model_bias(CUTE_LAYER_S1A))
#define m01a   (cute_model_multiplier(CUTE_LAYER_S1A))
#define s01a   (cute_model_shift(CUTE_LAYER_S1A))
#define w01b   (cute_model_weight(CUTE_LAYER_S1B))
#define b01b   (cute_model_bias(CUTE_LAYER_S1B))
#define m01b   (cute_model_multiplier(CUTE_LAYER_S1B))
#define s01b   (cute_model_shift(CUTE_LAYER_S1B))
#define w02a   (cute_model_weight(CUTE_LAYER_S2A))
#define b02a   (cute_model_bias(CUTE_LAYER_S2A))
#define m02a   (cute_model_multiplier(CUTE_LAYER_S2A))
#define s02a   (cute_model_shift(CUTE_LAYER_S2A))
#define w02b   (cute_model_weight(CUTE_LAYER_S2B))
#define b02b   (cute_model_bias(CUTE_LAYER_S2B))
#define m02b   (cute_model_multiplier(CUTE_LAYER_S2B))
#define s02b   (cute_model_shift(CUTE_LAYER_S2B))
#define w03a   (cute_model_weight(CUTE_LAYER_S3A))
#define b03a   (cute_model_bias(CUTE_LAYER_S3A))
#define m03a   (cute_model_multiplier(CUTE_LAYER_S3A))
#define s03a   (cute_model_shift(CUTE_LAYER_S3A))
#define w03b   (cute_model_weight(CUTE_LAYER_S3B))
#define b03b   (cute_model_bias(CUTE_LAYER_S3B))
#define m03b   (cute_model_multiplier(CUTE_LAYER_S3B))
#define s03b   (cute_model_shift(CUTE_LAYER_S3B))

#define w_h1a  (cute_model_weight(CUTE_LAYER_H1A))
#define b_h1a  (cute_model_bias(CUTE_LAYER_H1A))
#define m_h1a  (cute_model_multiplier(CUTE_LAYER_H1A))
#define s_h1a  (cute_model_shift(CUTE_LAYER_H1A))
#define w_h1dw (cute_model_weight(CUTE_LAYER_H1DW))
#define b_h1dw (cute_model_bias(CUTE_LAYER_H1DW))
#define m_h1dw (cute_model_multiplier(CUTE_LAYER_H1DW))
#define s_h1dw (cute_model_shift(CUTE_LAYER_H1DW))
#define w_h1pw (cute_model_weight(CUTE_LAYER_H1PW))
#define b_h1pw (cute_model_bias(CUTE_LAYER_H1PW))
#define m_h1pw (cute_model_multiplier(CUTE_LAYER_H1PW))
#define s_h1pw (cute_model_shift(CUTE_LAYER_H1PW))

#define w_h2a  (cute_model_weight(CUTE_LAYER_H2A))
#define b_h2a  (cute_model_bias(CUTE_LAYER_H2A))
#define m_h2a  (cute_model_multiplier(CUTE_LAYER_H2A))
#define s_h2a  (cute_model_shift(CUTE_LAYER_H2A))
#define w_h2dw (cute_model_weight(CUTE_LAYER_H2DW))
#define b_h2dw (cute_model_bias(CUTE_LAYER_H2DW))
#define m_h2dw (cute_model_multiplier(CUTE_LAYER_H2DW))
#define s_h2dw (cute_model_shift(CUTE_LAYER_H2DW))
#define w_h2pw (cute_model_weight(CUTE_LAYER_H2PW))
#define b_h2pw (cute_model_bias(CUTE_LAYER_H2PW))
#define m_h2pw (cute_model_multiplier(CUTE_LAYER_H2PW))
#define s_h2pw (cute_model_shift(CUTE_LAYER_H2PW))

#define w_h3a  (cute_model_weight(CUTE_LAYER_H3A))
#define b_h3a  (cute_model_bias(CUTE_LAYER_H3A))
#define m_h3a  (cute_model_multiplier(CUTE_LAYER_H3A))
#define s_h3a  (cute_model_shift(CUTE_LAYER_H3A))
#define w_h3dw (cute_model_weight(CUTE_LAYER_H3DW))
#define b_h3dw (cute_model_bias(CUTE_LAYER_H3DW))
#define m_h3dw (cute_model_multiplier(CUTE_LAYER_H3DW))
#define s_h3dw (cute_model_shift(CUTE_LAYER_H3DW))
#define w_h3pw (cute_model_weight(CUTE_LAYER_H3PW))
#define b_h3pw (cute_model_bias(CUTE_LAYER_H3PW))
#define m_h3pw (cute_model_multiplier(CUTE_LAYER_H3PW))
#define s_h3pw (cute_model_shift(CUTE_LAYER_H3PW))

#define w_h4a  (cute_model_weight(CUTE_LAYER_H4A))
#define b_h4a  (cute_model_bias(CUTE_LAYER_H4A))
#define m_h4a  (cute_model_multiplier(CUTE_LAYER_H4A))
#define s_h4a  (cute_model_shift(CUTE_LAYER_H4A))
#define w_h4dw (cute_model_weight(CUTE_LAYER_H4DW))
#define b_h4dw (cute_model_bias(CUTE_LAYER_H4DW))
#define m_h4dw (cute_model_multiplier(CUTE_LAYER_H4DW))
#define s_h4dw (cute_model_shift(CUTE_LAYER_H4DW))
#define w_h4pw (cute_model_weight(CUTE_LAYER_H4PW))
#define b_h4pw (cute_model_bias(CUTE_LAYER_H4PW))
#define m_h4pw (cute_model_multiplier(CUTE_LAYER_H4PW))
#define s_h4pw (cute_model_shift(CUTE_LAYER_H4PW))

#define w_h5a  (cute_model_weight(CUTE_LAYER_H5A))
#define b_h5a  (cute_model_bias(CUTE_LAYER_H5A))
#define m_h5a  (cute_model_multiplier(CUTE_LAYER_H5A))
#define s_h5a  (cute_model_shift(CUTE_LAYER_H5A))
#define w_h5dw (cute_model_weight(CUTE_LAYER_H5DW))
#define b_h5dw (cute_model_bias(CUTE_LAYER_H5DW))
#define m_h5dw (cute_model_multiplier(CUTE_LAYER_H5DW))
#define s_h5dw (cute_model_shift(CUTE_LAYER_H5DW))
#define w_h5pw (cute_model_weight(CUTE_LAYER_H5PW))
#define b_h5pw (cute_model_bias(CUTE_LAYER_H5PW))
#define m_h5pw (cute_model_multiplier(CUTE_LAYER_H5PW))
#define s_h5pw (cute_model_shift(CUTE_LAYER_H5PW))

#define w_head  (cute_model_weight(CUTE_LAYER_HEAD))
#define b_head  (cute_model_bias(CUTE_LAYER_HEAD))
#define m_head  (cute_model_multiplier(CUTE_LAYER_HEAD))
#define s_head  (cute_model_shift(CUTE_LAYER_HEAD))
