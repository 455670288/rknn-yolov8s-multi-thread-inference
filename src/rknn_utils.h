#pragma once

#include <stdio.h>
#include <stdlib.h>
#include <string>
#include <memory.h>
#include <assert.h>

#include <rknn_api.h>

#ifdef RK3399PRO
typedef struct _rknn_tensor_memory
{
    void *virt_addr;    /* the virtual address of tensor buffer. */
    uint64_t phys_addr; /* the physical address of tensor buffer. */
    int32_t fd;         /* the fd of tensor buffer. */
    int32_t offset;     /* indicates the offset of the memory. */
    uint32_t size;      /* the size of tensor buffer. */
    uint32_t flags;     /* the flags of tensor buffer, reserved */
    void *priv_data;    /* the private data of tensor buffer. */
} rknn_tensor_mem;
#endif

typedef enum
{
    NORMAL_API = 0,
    ZERO_COPY_API,
} API_TYPE;

typedef struct _RKNN_INPUT_PARAM
{
    /*
        RKNN_INPUT has follow param:
        index, buf, size, pass_through, fmt, type

        Here we keep:
            pass_through,
            'fmt' as 'layout_fmt',
            'type' as 'dtype'

        And add:
            api_type to record normal_api/ zero_copy_api
            enable to assign if this param was used
            _already_init to record if this param was already init
    */
    uint8_t pass_through;
    rknn_tensor_format layout_fmt;
    rknn_tensor_type dtype;

    API_TYPE api_type;
    bool enable = false;
    bool _already_init = false;
} RKNN_INPUT_PARAM;

typedef struct _RKNN_OUTPUT_PARAM
{
    uint8_t want_float;

    API_TYPE api_type;
    bool enable = false;
    bool _already_init = false;
} RKNN_OUTPUT_PARAM;

typedef struct _MODEL_INFO
{
    int npu_core = -1;

    // model load success or not
    bool load_success = false;

    // char* m_path = nullptr;
    std::string m_path = "";
    rknn_context ctx;
    bool is_dyn_shape = false;

    int n_input;
    rknn_tensor_attr *in_attr = nullptr;
    rknn_tensor_attr *in_attr_native = nullptr;
    rknn_input *inputs;
    rknn_tensor_mem **input_mem;
    RKNN_INPUT_PARAM *rknn_input_param;

    // bool inputs_alreay_init = false;

    int n_output;
    rknn_tensor_attr *out_attr = nullptr;
    rknn_tensor_attr *out_attr_native = nullptr;
    rknn_output *outputs;
    rknn_tensor_mem **output_mem;
    RKNN_OUTPUT_PARAM *rknn_output_param;

    // bool outputs_alreay_init = false;
    bool verbose_log = true;
    int diff_input_idx = -1;

    // memory could be set ousider
    int init_flag = 0;
#ifndef RKNN_NPU_1
    rknn_input_range *dyn_range;
    rknn_mem_size mem_size;
    rknn_tensor_mem *internal_mem_outside = nullptr;
    rknn_tensor_mem *internal_mem_max = nullptr;
#endif

} MODEL_INFO;

static void dump_tensor_attr(rknn_tensor_attr *attr);
// void dump_tensor_attr(rknn_tensor_attr *attr);
int rknn_util_get_type_size(rknn_tensor_type type);

int rknn_util_init(MODEL_INFO *model_info);
int rknn_util_init_share_weight(MODEL_INFO *model_info, MODEL_INFO *src_model_info);
int rknn_util_query_model_info(MODEL_INFO *model_info);

int rknn_util_init_input_buffer(MODEL_INFO *model_info, int node_index, API_TYPE api_type, uint8_t pass_through, rknn_tensor_type dtype, rknn_tensor_format layout_fmt);

int rknn_util_init_output_buffer(MODEL_INFO *model_info, int node_index, API_TYPE api_type, uint8_t want_float);

int rknn_util_init_input_buffer_all(MODEL_INFO *model_info, API_TYPE default_api_type, rknn_tensor_type default_t_type);

int rknn_util_init_output_buffer_all(MODEL_INFO *model_info, API_TYPE default_api_type, uint8_t default_want_float);

// order is: x -> model_top -> model_bottom -> result
int rknn_util_connect_models_node(MODEL_INFO *model_top, int top_out_index, MODEL_INFO *model_bottom, int bottom_in_index);

int rknn_util_release(MODEL_INFO *model_info);

// for native process
int offset_nchw_2_nc1hwc2(rknn_tensor_attr *src_attr, rknn_tensor_attr *native_attr, int offset, bool batch);

int rknn_util_query_dynamic_input(MODEL_INFO *model_info);

int rknn_util_reset_dynamic_input(MODEL_INFO *model_info, int dynamic_shape_group_index);

int rknn_util_thread_init(MODEL_INFO **model_infos, int num);