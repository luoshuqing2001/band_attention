#include <torch/extension.h>
#include "band_attention.h"

void torch_launch_band_attention(torch::Tensor &x,
                                torch::Tensor &attn,
                                torch::Tensor &q,
                                torch::Tensor &k,
                                torch::Tensor &v,
                                int64_t window,
                                int64_t bs,
                                int64_t nh,
                                int64_t nt,
                                int64_t channel) {
    launch_band_attention((float *)x.data_ptr(),
                        (float *)attn.data_ptr(),
                        (float *)q.data_ptr(),
                        (float *)k.data_ptr(),
                        (float *)v.data_ptr(),
                        window, bs, nh, nt, channel);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("torch_launch_band_attention",
          &torch_launch_band_attention,
          "band attention kernel warpper");
}

TORCH_LIBRARY(band_attention, m) {
    m.def("torch_launch_band_attention", torch_launch_band_attention);
}