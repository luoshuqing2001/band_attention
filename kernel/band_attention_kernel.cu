#define BLOCK_SIZE 10
#define FEATURE_SIZE 8
#define Delta 8 // prevent softmax overflow

typedef struct {
    int width;
    int height;
    int stride;
    float* elements;
} Matrix;

// Get a matrix element
__device__ float GetElementFromMatrix(const Matrix A, int row, int col)
{
    return A.elements[row * A.stride + col];
}

// Set a matrix element
__device__ void SetMatrixElement(Matrix A, int row, int col, float value)
{
    A.elements[row * A.stride + col] = value;
}

__device__ Matrix GetSubMatrixQ(float* q, int block_2, int block_3, int nh, int nt, int channel)
{
    Matrix Asub;
    Asub.width = channel;
    Asub.height = nt;
    Asub.stride = channel;
    Asub.elements = &q[block_2 * (nh * nt * channel) 
                               + block_3 * (nt * channel)];
    return Asub;
}

__device__ Matrix GetSubMatrixK(float* k, int block_2, int block_3, int nh, int nt, int channel)
{
    Matrix Asub;
    Asub.width = channel;
    Asub.height = nt;
    Asub.stride = channel;
    Asub.elements = &k[block_2 * (nh * nt * channel) 
                               + block_3 * (nt * channel)];
    return Asub;
}

__device__ Matrix GetSubMatrixAttn(float* attn, int block_2, int block_3, int nh, int nt)
{
    Matrix Asub;
    Asub.width = nt;
    Asub.height = nt;
    Asub.stride = nt;
    Asub.elements = &attn[block_2 * (nh * nt * nt) 
                                  + block_3 * (nt * nt)];
    return Asub;
}

__device__ Matrix GetSubMatrixLeft(Matrix A, int row, int col, int channel)
{
    Matrix Asub;
    Asub.width = FEATURE_SIZE;
    Asub.height = BLOCK_SIZE;
    Asub.stride = channel;
    Asub.elements = &A.elements[channel * row 
                                        + FEATURE_SIZE * col];
    return Asub;
}

__device__ Matrix GetSubMatrixRight(Matrix A, int row, int col, int window, int nt, int channel)
{
    Matrix Asub;
    Asub.width = FEATURE_SIZE;
    if ((row >= window) && (row < nt - window)) {
        Asub.height = 1 + 2 * window;
    }
    else if (row < window) {
        Asub.height = 1 + window + row;
    }
    else {
        Asub.height = window + nt - row;
    }
    Asub.stride = channel;
    Asub.elements = &A.elements[channel * row 
                                        + FEATURE_SIZE * col];
    return Asub;
}


__global__ void band_attention_kernel(float* attn,
                                      float* q,
                                      float* k,
                                      int window,
                                      int nh,
                                      int nt,
                                      int channel) {
    // Block axis
    int block_1 = blockIdx.x;
    int block_2 = blockIdx.y;
    int block_3 = blockIdx.z;

    Matrix Q_sub = GetSubMatrixQ(q, block_2, block_3, nh, nt, channel);
    Matrix K_sub = GetSubMatrixK(k, block_2, block_3, nh, nt, channel);
    Matrix Attn_sub = GetSubMatrixAttn(attn, block_2, block_3, nh, nt);

    // thread axis
    int thread_1 = threadIdx.x;
    int thread_2 = threadIdx.y;
    int thread_3 = threadIdx.z;

    float Attn_value = 0.0;
    int row = thread_1 + block_1 * BLOCK_SIZE;

    int prefix_shift = 0, delta_shift = thread_1;
    if (row < window) {
        if (row - window + thread_2 < 0)
            return;
        else {
            prefix_shift = window - row;
            delta_shift = 0;
        }
    } else if (row < BLOCK_SIZE) {
        delta_shift = thread_1 - window;
    } else if (row >= nt - window) {
        if (row - window + thread_2 > nt - 1)
            return;
    }

    for (int m = 0; m < (channel / FEATURE_SIZE); ++m) 
    {
        Matrix QQ_sub = GetSubMatrixLeft(Q_sub, block_1 * BLOCK_SIZE, m, channel);
        Matrix KK_sub = GetSubMatrixRight(K_sub, max(block_1 * BLOCK_SIZE - window, 0), m, window, nt, channel);

        __shared__ float Qs[BLOCK_SIZE][FEATURE_SIZE];
        __shared__ float Ks[BLOCK_SIZE * 3][FEATURE_SIZE];

        Qs[thread_1][thread_3] = GetElementFromMatrix(QQ_sub, thread_1, thread_3);
        Ks[thread_2 - prefix_shift + delta_shift][thread_3] = GetElementFromMatrix(KK_sub, thread_2 - prefix_shift + delta_shift, thread_3);

        __syncthreads();

        for (int e = 0; e < FEATURE_SIZE; ++e)
            Attn_value += Qs[thread_1][e] * Ks[thread_2 - prefix_shift + delta_shift][e];

        __syncthreads();
    }

    SetMatrixElement(Attn_sub, row, row - window + thread_2, Attn_value);
}

__global__ void softmax_kernel(float* attn, 
                               int window,
                               int nh,
                               int nt) {
    // Block axis
    int block_1 = blockIdx.x;
    int block_2 = blockIdx.y;
    int block_3 = blockIdx.z;
    
    // thread axis
    int thread_1 = threadIdx.x;
    
    Matrix Attn_sub = GetSubMatrixAttn(attn, block_2, block_3, nh, nt);

    int row = thread_1 + block_1 * BLOCK_SIZE;

    int start_idx = 0;
    if (row - window > 0)
        start_idx = row - window;
    int end_idx = nt - 1;
    if (row + window < nt - 1)
        end_idx = row + window;

    float sum_value = 0.0;
    for (int idx = start_idx; idx <= end_idx; ++idx) {
        sum_value += exp(GetElementFromMatrix(Attn_sub, row, idx) - Delta);
    }

    for (int idx = start_idx; idx <= end_idx; ++idx) {
        float tmp = GetElementFromMatrix(Attn_sub, row, idx) - Delta - log(sum_value);
        SetMatrixElement(Attn_sub, row, idx, exp(tmp));
    }
}

__global__ void attention_v_kernel(float* x, 
                                   float* attn, 
                                   float* v, 
                                   int window, 
                                   int nh, 
                                   int nt, 
                                   int channel) {
    // Block axis
    int block_1 = blockIdx.x;
    int block_2 = blockIdx.y;
    int block_3 = blockIdx.z;

    Matrix X_sub = GetSubMatrixQ(x, block_2, block_3, nh, nt, channel);
    Matrix V_sub = GetSubMatrixK(v, block_2, block_3, nh, nt, channel);
    Matrix Attn_sub = GetSubMatrixAttn(attn, block_2, block_3, nh, nt);

    // thread axis
    int thread_1 = threadIdx.x;
    int thread_2 = threadIdx.y;
    int thread_3 = threadIdx.z;

    float x_result = 0.0;
    int row = thread_1 + block_1 * BLOCK_SIZE;
    int prefix_shift = 0, delta_shift = thread_1;
    if (row < window) {
        if (row - window + thread_2 < 0)
            return;
        else {
            prefix_shift = window - row;
            delta_shift = 0;
        }
    } else if (row < BLOCK_SIZE) {
        delta_shift = thread_1 - window;
    } else if (row >= nt - window) {
        if (row - window + thread_2 > nt - 1)
            return;
    }

    int start_idx = 0;
    if (row - window > 0)
        start_idx = row - window;
    int end_idx = nt - 1;
    if (row + window < nt - 1)
        end_idx = row + window;

    for (int m = 0; m < (channel / FEATURE_SIZE); ++m) 
    {
        Matrix VV_sub = GetSubMatrixRight(V_sub, max(block_1 * BLOCK_SIZE - window, 0), m, window, nt, channel);

        __shared__ float Vs[BLOCK_SIZE * 3][FEATURE_SIZE];
        __shared__ float Attns[BLOCK_SIZE][BLOCK_SIZE * 3];

        Vs[thread_2 - prefix_shift + delta_shift][thread_3] = GetElementFromMatrix(VV_sub, thread_2 - prefix_shift + delta_shift, thread_3);
        Attns[thread_1][thread_2 - prefix_shift + delta_shift] = GetElementFromMatrix(Attn_sub, row, thread_2 - prefix_shift + delta_shift + max(block_1 * BLOCK_SIZE - window, 0));

        __syncthreads();

        float tmp = 0.0;
        for (int idx = start_idx; idx <= end_idx; ++idx)
            tmp += Attns[thread_1][idx - max(block_1 * BLOCK_SIZE - window, 0)] * Vs[idx - max(block_1 * BLOCK_SIZE - window, 0)][thread_3];

        SetMatrixElement(X_sub, row, m * FEATURE_SIZE + thread_3, tmp);

        __syncthreads();
    }
}

void launch_band_attention(float* x,
                           float* attn,
                           float* q,
                           float* k,
                           float* v,
                           int window,
                           int bs,
                           int nh,
                           int nt,
                           int channel) {
    // data has already on the device side
    // Q @ K
    dim3 DimBlock(BLOCK_SIZE, 1 + 2 * window, FEATURE_SIZE);
    dim3 DimGrid(nt / BLOCK_SIZE, bs, nh);
    band_attention_kernel<<<DimGrid, DimBlock>>>(attn, q, k, window, nh, nt, channel);

    // Softmax
    dim3 DimBlock_2(BLOCK_SIZE);
    dim3 DimGrid_2(nt / BLOCK_SIZE, bs, nh);
    softmax_kernel<<<DimGrid_2, DimBlock_2>>>(attn, window, nh, nt);

    // Attn @ V
    dim3 DimBlock_3(BLOCK_SIZE, 1 + 2 * window, FEATURE_SIZE);
    dim3 DimGrid_3(nt / BLOCK_SIZE, bs, nh);
    attention_v_kernel<<<DimGrid_3, DimBlock_3>>>(x, attn, v, window, nh, nt, channel);
}

