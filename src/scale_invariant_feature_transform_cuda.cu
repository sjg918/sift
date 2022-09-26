
#ifndef CUDAUTILS
#define CUDAUTILS

#include <cstdio>
#include <iostream>
#include <algorithm>
#include <cmath>

#ifdef WIN32
#include <intrin.h>
#endif

#define safeCall(err)       __safeCall(err, __FILE__, __LINE__)
#define safeThreadSync()    __safeThreadSync(__FILE__, __LINE__)
#define checkMsg(msg)       __checkMsg(msg, __FILE__, __LINE__)


inline void __safeCall(cudaError err, const char* file, const int line)
{
    if (cudaSuccess != err) {
        fprintf(stderr, "safeCall() Runtime API error in file <%s>, line %i : %s.\n", file, line, cudaGetErrorString(err));
        exit(-1);
    }
}


inline void __safeThreadSync(const char* file, const int line)
{
    cudaError err = cudaDeviceSynchronize();
    if (cudaSuccess != err) {
        fprintf(stderr, "threadSynchronize() Driver API error in file '%s' in line %i : %s.\n", file, line, cudaGetErrorString(err));
        exit(-1);
    }
}


inline void __checkMsg(const char* errorMessage, const char* file, const int line)
{
    cudaError_t err = cudaGetLastError();
    if (cudaSuccess != err) {
        fprintf(stderr, "checkMsg() CUDA error: %s in file <%s>, line %i : %s.\n", errorMessage, file, line, cudaGetErrorString(err));
        exit(-1);
    }
}


inline bool deviceInit(int dev)
{
    int deviceCount;
    safeCall(cudaGetDeviceCount(&deviceCount));
    if (deviceCount == 0) {
        fprintf(stderr, "CUDA error: no devices supporting CUDA.\n");
        return false;
    }
    if (dev < 0) dev = 0;
    if (dev > deviceCount - 1) dev = deviceCount - 1;
    cudaDeviceProp deviceProp;
    safeCall(cudaGetDeviceProperties(&deviceProp, dev));
    if (deviceProp.major < 1) {
        fprintf(stderr, "error: device does not support CUDA.\n");
        return false;
    }
    safeCall(cudaSetDevice(dev));
    return true;
}


template <class T>
__device__ __inline__ T ShiftDown(T var, unsigned int delta, int width = 32) {
#if (CUDART_VERSION >= 9000)
    return __shfl_down_sync(0xffffffff, var, delta, width);
#else
    return __shfl_down(var, delta, width);
#endif
}

template <class T>
__device__ __inline__ T ShiftUp(T var, unsigned int delta, int width = 32) {
#if (CUDART_VERSION >= 9000)
    return __shfl_up_sync(0xffffffff, var, delta, width);
#else
    return __shfl_up(var, delta, width);
#endif
}

template <class T>
__device__ __inline__ T Shuffle(T var, unsigned int lane, int width = 32) {
#if (CUDART_VERSION >= 9000)
    return __shfl_sync(0xffffffff, var, lane, width);
#else
    return __shfl(var, lane, width);
#endif
}


#endif

#ifndef CUDASIFTD
#define CUDASIFTD

#define NUM_SCALES      5

// Scale down thread block width
#define SCALEDOWN_W    64 // 60 

// Scale down thread block height
#define SCALEDOWN_H    16 // 8

// Scale up thread block width
#define SCALEUP_W      64

// Scale up thread block height
#define SCALEUP_H       8

// Find point thread block width
#define MINMAX_W       30 //32 

// Find point thread block height
#define MINMAX_H        8 //16 

// Laplace thread block width
#define LAPLACE_W     128 // 56

// Laplace rows per thread
#define LAPLACE_H       4

// Number of laplace scales
#define LAPLACE_S   (NUM_SCALES+3)

// Laplace filter kernel radius
#define LAPLACE_R       4

#define LOWPASS_W      24 //56
#define LOWPASS_H      32 //16
#define LOWPASS_R       4

//====================== Number of threads ====================//
// ScaleDown:               SCALEDOWN_W + 4
// LaplaceMulti:            (LAPLACE_W+2*LAPLACE_R)*LAPLACE_S
// FindPointsMulti:         MINMAX_W + 2
// ComputeOrientations:     128
// ExtractSiftDescriptors:  256

//====================== Number of blocks ====================//
// ScaleDown:               (width/SCALEDOWN_W) * (height/SCALEDOWN_H)
// LaplceMulti:             (width+2*LAPLACE_R)/LAPLACE_W * height
// FindPointsMulti:         (width/MINMAX_W)*NUM_SCALES * (height/MINMAX_H)
// ComputeOrientations:     numpts
// ExtractSiftDescriptors:  numpts

#endif

__constant__ int d_MaxNumPoints;
__device__ unsigned int d_PointCounter[8 * 2 + 1];
__constant__ float d_ScaleDownKernel[5];
__constant__ float d_LowPassKernel[2 * LOWPASS_R + 1];
__constant__ float d_LaplaceKernel[8 * 12 * 16];

int iDivUp(int a, int b) { return (a % b != 0) ? (a / b + 1) : (a / b); }
int iDivDown(int a, int b) { return a / b; }
int iAlignUp(int a, int b) { return (a % b != 0) ? (a - a % b + b) : a; }
int iAlignDown(int a, int b) { return a - a % b; }

#ifndef CUDASIFT_H
#define CUDASIFT_H

typedef struct {
    float xpos;
    float ypos;
    float scale;
    float sharpness;
    float edgeness;
    float orientation;
    float score;
    float ambiguity;
    int match;
    float match_xpos;
    float match_ypos;
    float match_error;
    float subsampling;
    float empty[3];
    float data[128];
} SiftPoint;

typedef struct {
    int numPts;         // Number of available Sift points
    int maxPts;         // Number of allocated Sift points
#ifdef MANAGEDMEM
    SiftPoint* m_data;  // Managed data
#else
    SiftPoint* h_data;  // Host (CPU) data
    SiftPoint* d_data;  // Device (GPU) data
#endif
} SiftData;

void InitSiftData(SiftData& data, int num, bool host, bool dev)
{
    data.numPts = 0;
    data.maxPts = num;
    int sz = sizeof(SiftPoint) * num;
#ifdef MANAGEDMEM
    safeCall(cudaMallocManaged((void**)&data.m_data, sz));
#else
    data.h_data = NULL;
    if (host)
        data.h_data = (SiftPoint*)malloc(sz);
    data.d_data = NULL;
    if (dev)
        safeCall(cudaMalloc((void**)&data.d_data, sz));
#endif
}

void FreeSiftData(SiftData& data)
{
#ifdef MANAGEDMEM
    safeCall(cudaFree(data.m_data));
#else
    if (data.d_data != NULL)
        safeCall(cudaFree(data.d_data));
    data.d_data = NULL;
    if (data.h_data != NULL)
        free(data.h_data);
#endif
    data.numPts = 0;
    data.maxPts = 0;
}

float* AllocSiftTempMemory(int width, int height, int numOctaves, bool scaleUp)
{
    const int nd = NUM_SCALES + 3;
    int w = width * (scaleUp ? 2 : 1);
    int h = height * (scaleUp ? 2 : 1);
    int p = iAlignUp(w, 128);
    int size = h * p;                 // image sizes
    int sizeTmp = nd * h * p;           // laplace buffer sizes
    for (int i = 0; i < numOctaves; i++) {
        w /= 2;
        h /= 2;
        int p = iAlignUp(w, 128);
        size += h * p;
        sizeTmp += nd * h * p;
    }
    float* memoryTmp = NULL;
    size_t pitch;
    size += sizeTmp;
    safeCall(cudaMallocPitch((void**)&memoryTmp, &pitch, (size_t)4096, (size + 4095) / 4096 * sizeof(float)));
#ifdef VERBOSE
    printf("Allocated memory size: %d bytes\n", size);
    printf("Memory allocation time =      %.2f ms\n\n", timer.read());
#endif
    return memoryTmp;
}

void FreeSiftTempMemory(float* memoryTmp)
{
    if (memoryTmp)
        safeCall(cudaFree(memoryTmp));
}

#endif

#ifndef CUDAIMAGE
#define CUDAIMAGE

class CudaImage {
public:
    int width, height;
    int pitch;
    float* h_data;
    float* d_data;
    float* t_data;
    bool d_internalAlloc;
    bool h_internalAlloc;
public:
    CudaImage();
    ~CudaImage();
    void Allocate(int width, int height, int pitch, bool withHost, float* devMem = NULL, float* hostMem = NULL);
    double Download();
    double Readback();
    double InitTexture();
    double CopyToTexture(CudaImage& dst, bool host);
};

void CudaImage::Allocate(int w, int h, int p, bool host, float* devmem, float* hostmem)
{
    width = w;
    height = h;
    pitch = p;
    d_data = devmem;
    h_data = hostmem;
    t_data = NULL;
    if (devmem == NULL) {
        safeCall(cudaMallocPitch((void**)&d_data, (size_t*)&pitch, (size_t)(sizeof(float) * width), (size_t)height));
        pitch /= sizeof(float);
        if (d_data == NULL)
            printf("Failed to allocate device data\n");
        d_internalAlloc = true;
    }
    if (host && hostmem == NULL) {
        h_data = (float*)malloc(sizeof(float) * pitch * height);
        h_internalAlloc = true;
    }
}

CudaImage::CudaImage() :
    width(0), height(0), d_data(NULL), h_data(NULL), t_data(NULL), d_internalAlloc(false), h_internalAlloc(false)
{

}

CudaImage::~CudaImage()
{
    if (d_internalAlloc && d_data != NULL)
        safeCall(cudaFree(d_data));
    d_data = NULL;
    if (h_internalAlloc && h_data != NULL)
        free(h_data);
    h_data = NULL;
    if (t_data != NULL)
        safeCall(cudaFreeArray((cudaArray*)t_data));
    t_data = NULL;
}

double CudaImage::Download()
{
    int p = sizeof(float) * pitch;
    if (d_data != NULL && h_data != NULL)
        safeCall(cudaMemcpy2D(d_data, p, h_data, sizeof(float) * width, sizeof(float) * width, height, cudaMemcpyHostToDevice));
#ifdef VERBOSE
    printf("Download time =               %.2f ms\n", gpuTime);
#endif
    return 0.0;
}

double CudaImage::Readback()
{
    int p = sizeof(float) * pitch;
    safeCall(cudaMemcpy2D(h_data, sizeof(float) * width, d_data, p, sizeof(float) * width, height, cudaMemcpyDeviceToHost));
#ifdef VERBOSE
    printf("Readback time =               %.2f ms\n", gpuTime);
#endif
    return 0.0;
}

double CudaImage::InitTexture()
{
    cudaChannelFormatDesc t_desc = cudaCreateChannelDesc<float>();
    safeCall(cudaMallocArray((cudaArray**)&t_data, &t_desc, pitch, height));
    if (t_data == NULL)
        printf("Failed to allocated texture data\n");
#ifdef VERBOSE
    printf("InitTexture time =            %.2f ms\n", gpuTime);
#endif
    return 0.0;
}

double CudaImage::CopyToTexture(CudaImage& dst, bool host)
{
    if (dst.t_data == NULL) {
        printf("Error CopyToTexture: No texture data\n");
        return 0.0;
    }
    if ((!host || h_data == NULL) && (host || d_data == NULL)) {
        printf("Error CopyToTexture: No source data\n");
        return 0.0;
    }
    if (host)
        safeCall(cudaMemcpyToArray((cudaArray*)dst.t_data, 0, 0, h_data, sizeof(float) * pitch * dst.height, cudaMemcpyHostToDevice));
    else
        safeCall(cudaMemcpyToArray((cudaArray*)dst.t_data, 0, 0, d_data, sizeof(float) * pitch * dst.height, cudaMemcpyDeviceToDevice));
    safeCall(cudaDeviceSynchronize());
#ifdef VERBOSE
    printf("CopyToTexture time =          %.2f ms\n", gpuTime);
#endif
    return 0.0;
}
#endif

#ifndef FUNCS
#define FUNCS

int ExtractSiftLoop(SiftData& siftData, CudaImage& img, int numOctaves, double initBlur, float thresh, float lowestScale, float subsampling, float* memoryTmp, float* memorySub);
void ExtractSiftOctave(SiftData& siftData, CudaImage& img, int octave, float thresh, float lowestScale, float subsampling, float* memoryTmp);
double ScaleDown(CudaImage& res, CudaImage& src, float variance);
double ScaleUp(CudaImage& res, CudaImage& src);
double ComputeOrientations(cudaTextureObject_t texObj, CudaImage& src, SiftData& siftData, int octave);
double ExtractSiftDescriptors(cudaTextureObject_t texObj, SiftData& siftData, float subsampling, int octave);
double OrientAndExtract(cudaTextureObject_t texObj, SiftData& siftData, float subsampling, int octave);
double RescalePositions(SiftData& siftData, float scale);
double LowPass(CudaImage& res, CudaImage& src, float scale);
void PrepareLaplaceKernels(int numOctaves, float initBlur, float* kernel);
double LaplaceMulti(cudaTextureObject_t texObj, CudaImage& baseImage, CudaImage* results, int octave);
double FindPointsMulti(CudaImage* sources, SiftData& siftData, float thresh, float edgeLimit, float factor, float lowestScale, float subsampling, int octave);

__device__ float FastAtan2(float y, float x)
{
    float absx = abs(x);
    float absy = abs(y);
    float a = __fdiv_rn(min(absx, absy), max(absx, absy));
    float s = a * a;
    float r = ((-0.0464964749f * s + 0.15931422f) * s - 0.327622764f) * s * a + a;
    r = (absy > absx ? 1.57079637f - r : r);
    r = (x < 0 ? 3.14159274f - r : r);
    r = (y < 0 ? -r : r);
    return r;
}

__global__ void LowPassBlock_kernel(float* d_Image, float* d_Result, int width, int pitch, int height)
{
    __shared__ float xrows[16][32];
    const int tx = threadIdx.x;
    const int ty = threadIdx.y;
    const int xp = blockIdx.x * LOWPASS_W + tx;
    const int yp = blockIdx.y * LOWPASS_H + ty;
    const int N = 16;
    float* k = d_LowPassKernel;
    int xl = max(min(xp - 4, width - 1), 0);
#pragma unroll
    for (int l = -8; l < 4; l += 4) {
        int ly = l + ty;
        int yl = max(min(yp + l + 4, height - 1), 0);
        float val = d_Image[yl * pitch + xl];
        val = k[4] * ShiftDown(val, 4) +
            k[3] * (ShiftDown(val, 5) + ShiftDown(val, 3)) +
            k[2] * (ShiftDown(val, 6) + ShiftDown(val, 2)) +
            k[1] * (ShiftDown(val, 7) + ShiftDown(val, 1)) +
            k[0] * (ShiftDown(val, 8) + val);
        xrows[ly + 8][tx] = val;
    }
    __syncthreads();
#pragma unroll
    for (int l = 4; l < LOWPASS_H; l += 4) {
        int ly = l + ty;
        int yl = min(yp + l + 4, height - 1);
        float val = d_Image[yl * pitch + xl];
        val = k[4] * ShiftDown(val, 4) +
            k[3] * (ShiftDown(val, 5) + ShiftDown(val, 3)) +
            k[2] * (ShiftDown(val, 6) + ShiftDown(val, 2)) +
            k[1] * (ShiftDown(val, 7) + ShiftDown(val, 1)) +
            k[0] * (ShiftDown(val, 8) + val);
        xrows[(ly + 8) % N][tx] = val;
        int ys = yp + l - 4;
        if (xp < width && ys < height && tx < LOWPASS_W)
            d_Result[ys * pitch + xp] = k[4] * xrows[(ly + 0) % N][tx] +
            k[3] * (xrows[(ly - 1) % N][tx] + xrows[(ly + 1) % N][tx]) +
            k[2] * (xrows[(ly - 2) % N][tx] + xrows[(ly + 2) % N][tx]) +
            k[1] * (xrows[(ly - 3) % N][tx] + xrows[(ly + 3) % N][tx]) +
            k[0] * (xrows[(ly - 4) % N][tx] + xrows[(ly + 4) % N][tx]);
        __syncthreads();
    }
    int ly = LOWPASS_H + ty;
    int ys = yp + LOWPASS_H - 4;
    if (xp < width && ys < height && tx < LOWPASS_W)
        d_Result[ys * pitch + xp] = k[4] * xrows[(ly + 0) % N][tx] +
        k[3] * (xrows[(ly - 1) % N][tx] + xrows[(ly + 1) % N][tx]) +
        k[2] * (xrows[(ly - 2) % N][tx] + xrows[(ly + 2) % N][tx]) +
        k[1] * (xrows[(ly - 3) % N][tx] + xrows[(ly + 3) % N][tx]) +
        k[0] * (xrows[(ly - 4) % N][tx] + xrows[(ly + 4) % N][tx]);
}

__global__ void ScaleDown_kernel(float* d_Result, float* d_Data, int width, int pitch, int height, int newpitch)
{
    __shared__ float inrow[SCALEDOWN_W + 4];
    __shared__ float brow[5 * (SCALEDOWN_W / 2)];
    __shared__ int yRead[SCALEDOWN_H + 4];
    __shared__ int yWrite[SCALEDOWN_H + 4];
#define dx2 (SCALEDOWN_W/2)
    const int tx = threadIdx.x;
    const int tx0 = tx + 0 * dx2;
    const int tx1 = tx + 1 * dx2;
    const int tx2 = tx + 2 * dx2;
    const int tx3 = tx + 3 * dx2;
    const int tx4 = tx + 4 * dx2;
    const int xStart = blockIdx.x * SCALEDOWN_W;
    const int yStart = blockIdx.y * SCALEDOWN_H;
    const int xWrite = xStart / 2 + tx;
    float k0 = d_ScaleDownKernel[0];
    float k1 = d_ScaleDownKernel[1];
    float k2 = d_ScaleDownKernel[2];
    if (tx < SCALEDOWN_H + 4) {
        int y = yStart + tx - 2;
        y = (y < 0 ? 0 : y);
        y = (y >= height ? height - 1 : y);
        yRead[tx] = y * pitch;
        yWrite[tx] = (yStart + tx - 4) / 2 * newpitch;
    }
    __syncthreads();
    int xRead = xStart + tx - 2;
    xRead = (xRead < 0 ? 0 : xRead);
    xRead = (xRead >= width ? width - 1 : xRead);

    int maxtx = min(dx2, width / 2 - xStart / 2);
    for (int dy = 0; dy < SCALEDOWN_H + 4; dy += 5) {
        {
            inrow[tx] = d_Data[yRead[dy + 0] + xRead];
            __syncthreads();
            if (tx < maxtx) {
                brow[tx4] = k0 * (inrow[2 * tx] + inrow[2 * tx + 4]) + k1 * (inrow[2 * tx + 1] + inrow[2 * tx + 3]) + k2 * inrow[2 * tx + 2];
                if (dy >= 4 && !(dy & 1))
                    d_Result[yWrite[dy + 0] + xWrite] = k2 * brow[tx2] + k0 * (brow[tx0] + brow[tx4]) + k1 * (brow[tx1] + brow[tx3]);
            }
            __syncthreads();
        }
        if (dy < (SCALEDOWN_H + 3)) {
            inrow[tx] = d_Data[yRead[dy + 1] + xRead];
            __syncthreads();
            if (tx < maxtx) {
                brow[tx0] = k0 * (inrow[2 * tx] + inrow[2 * tx + 4]) + k1 * (inrow[2 * tx + 1] + inrow[2 * tx + 3]) + k2 * inrow[2 * tx + 2];
                if (dy >= 3 && (dy & 1))
                    d_Result[yWrite[dy + 1] + xWrite] = k2 * brow[tx3] + k0 * (brow[tx1] + brow[tx0]) + k1 * (brow[tx2] + brow[tx4]);
            }
            __syncthreads();
        }
        if (dy < (SCALEDOWN_H + 2)) {
            inrow[tx] = d_Data[yRead[dy + 2] + xRead];
            __syncthreads();
            if (tx < maxtx) {
                brow[tx1] = k0 * (inrow[2 * tx] + inrow[2 * tx + 4]) + k1 * (inrow[2 * tx + 1] + inrow[2 * tx + 3]) + k2 * inrow[2 * tx + 2];
                if (dy >= 2 && !(dy & 1))
                    d_Result[yWrite[dy + 2] + xWrite] = k2 * brow[tx4] + k0 * (brow[tx2] + brow[tx1]) + k1 * (brow[tx3] + brow[tx0]);
            }
            __syncthreads();
        }
        if (dy < (SCALEDOWN_H + 1)) {
            inrow[tx] = d_Data[yRead[dy + 3] + xRead];
            __syncthreads();
            if (tx < maxtx) {
                brow[tx2] = k0 * (inrow[2 * tx] + inrow[2 * tx + 4]) + k1 * (inrow[2 * tx + 1] + inrow[2 * tx + 3]) + k2 * inrow[2 * tx + 2];
                if (dy >= 1 && (dy & 1))
                    d_Result[yWrite[dy + 3] + xWrite] = k2 * brow[tx0] + k0 * (brow[tx3] + brow[tx2]) + k1 * (brow[tx4] + brow[tx1]);
            }
            __syncthreads();
        }
        if (dy < SCALEDOWN_H) {
            inrow[tx] = d_Data[yRead[dy + 4] + xRead];
            __syncthreads();
            if (tx < dx2 && xWrite < width / 2) {
                brow[tx3] = k0 * (inrow[2 * tx] + inrow[2 * tx + 4]) + k1 * (inrow[2 * tx + 1] + inrow[2 * tx + 3]) + k2 * inrow[2 * tx + 2];
                if (!(dy & 1))
                    d_Result[yWrite[dy + 4] + xWrite] = k2 * brow[tx1] + k0 * (brow[tx4] + brow[tx3]) + k1 * (brow[tx0] + brow[tx2]);
            }
            __syncthreads();
        }
    }
}

__global__ void LaplaceMultiMem_kernel(float* d_Image, float* d_Result, int width, int pitch, int height, int octave)
{
    __shared__ float buff[(LAPLACE_W + 2 * LAPLACE_R) * LAPLACE_S];
    const int tx = threadIdx.x;
    const int xp = blockIdx.x * LAPLACE_W + tx;
    const int yp = blockIdx.y;
    float* data = d_Image + max(min(xp - LAPLACE_R, width - 1), 0);
    float temp[2 * LAPLACE_R + 1], kern[LAPLACE_S][LAPLACE_R + 1];
    if (xp < (width + 2 * LAPLACE_R)) {
        for (int i = 0; i <= 2 * LAPLACE_R; i++)
            temp[i] = data[max(0, min(yp + i - LAPLACE_R, height - 1)) * pitch];
        for (int scale = 0; scale < LAPLACE_S; scale++) {
            float* buf = buff + (LAPLACE_W + 2 * LAPLACE_R) * scale;
            float* kernel = d_LaplaceKernel + octave * 12 * 16 + scale * 16;
            for (int i = 0; i <= LAPLACE_R; i++)
                kern[scale][i] = kernel[i];
            float sum = kern[scale][0] * temp[LAPLACE_R];
#pragma unroll      
            for (int j = 1; j <= LAPLACE_R; j++)
                sum += kern[scale][j] * (temp[LAPLACE_R - j] + temp[LAPLACE_R + j]);
            buf[tx] = sum;
        }
    }
    __syncthreads();
    if (tx < LAPLACE_W && xp < width) {
        int scale = 0;
        float oldRes = kern[scale][0] * buff[tx + LAPLACE_R];
#pragma unroll
        for (int j = 1; j <= LAPLACE_R; j++)
            oldRes += kern[scale][j] * (buff[tx + LAPLACE_R - j] + buff[tx + LAPLACE_R + j]);
        for (int scale = 1; scale < LAPLACE_S; scale++) {
            float* buf = buff + (LAPLACE_W + 2 * LAPLACE_R) * scale;
            float res = kern[scale][0] * buf[tx + LAPLACE_R];
#pragma unroll
            for (int j = 1; j <= LAPLACE_R; j++)
                res += kern[scale][j] * (buf[tx + LAPLACE_R - j] + buf[tx + LAPLACE_R + j]);
            d_Result[(scale - 1) * height * pitch + yp * pitch + xp] = res - oldRes;
            oldRes = res;
        }
    }
}

__global__ void FindPointsMultiNew_kernel(float* d_Data0, SiftPoint* d_Sift, int width, int pitch, int height, float subsampling, float lowestScale, float thresh, float factor, float edgeLimit, int octave)
{
#define MEMWID (MINMAX_W + 2)
    __shared__ unsigned short points[2 * MEMWID];

    if (blockIdx.x == 0 && blockIdx.y == 0 && threadIdx.x == 0) {
        atomicMax(&d_PointCounter[2 * octave + 0], d_PointCounter[2 * octave - 1]);
        atomicMax(&d_PointCounter[2 * octave + 1], d_PointCounter[2 * octave - 1]);
    }
    int tx = threadIdx.x;
    int block = blockIdx.x / NUM_SCALES;
    int scale = blockIdx.x - NUM_SCALES * block;
    int minx = block * MINMAX_W;
    int maxx = min(minx + MINMAX_W, width);
    int xpos = minx + tx;
    int size = pitch * height;
    int ptr = size * scale + max(min(xpos - 1, width - 1), 0);

    int yloops = min(height - MINMAX_H * blockIdx.y, MINMAX_H);
    float maxv = 0.0f;
    for (int y = 0; y < yloops; y++) {
        int ypos = MINMAX_H * blockIdx.y + y;
        int yptr1 = ptr + ypos * pitch;
        float val = d_Data0[yptr1 + 1 * size];
        maxv = fmaxf(maxv, fabs(val));
    }
    //if (tx==0) printf("XXX1\n");
    if (!__any_sync(0xffffffff, maxv > thresh))
        return;
    //if (tx==0) printf("XXX2\n");

    int ptbits = 0;
    for (int y = 0; y < yloops; y++) {

        int ypos = MINMAX_H * blockIdx.y + y;
        int yptr1 = ptr + ypos * pitch;
        float d11 = d_Data0[yptr1 + 1 * size];
        if (__any_sync(0xffffffff, fabs(d11) > thresh)) {

            int yptr0 = ptr + max(0, ypos - 1) * pitch;
            int yptr2 = ptr + min(height - 1, ypos + 1) * pitch;
            float d01 = d_Data0[yptr1];
            float d10 = d_Data0[yptr0 + 1 * size];
            float d12 = d_Data0[yptr2 + 1 * size];
            float d21 = d_Data0[yptr1 + 2 * size];

            float d00 = d_Data0[yptr0];
            float d02 = d_Data0[yptr2];
            float ymin1 = fminf(fminf(d00, d01), d02);
            float ymax1 = fmaxf(fmaxf(d00, d01), d02);
            float d20 = d_Data0[yptr0 + 2 * size];
            float d22 = d_Data0[yptr2 + 2 * size];
            float ymin3 = fminf(fminf(d20, d21), d22);
            float ymax3 = fmaxf(fmaxf(d20, d21), d22);
            float ymin2 = fminf(fminf(ymin1, fminf(fminf(d10, d12), d11)), ymin3);
            float ymax2 = fmaxf(fmaxf(ymax1, fmaxf(fmaxf(d10, d12), d11)), ymax3);

            float nmin2 = fminf(ShiftUp(ymin2, 1), ShiftDown(ymin2, 1));
            float nmax2 = fmaxf(ShiftUp(ymax2, 1), ShiftDown(ymax2, 1));
            float minv = fminf(fminf(nmin2, ymin1), ymin3);
            minv = fminf(fminf(minv, d10), d12);
            float maxv = fmaxf(fmaxf(nmax2, ymax1), ymax3);
            maxv = fmaxf(fmaxf(maxv, d10), d12);

            if (tx > 0 && tx < MINMAX_W + 1 && xpos <= maxx)
                ptbits |= ((d11 < fminf(-thresh, minv)) | (d11 > fmaxf(thresh, maxv))) << y;
        }
    }

    unsigned int totbits = __popc(ptbits);
    unsigned int numbits = totbits;
    for (int d = 1; d < 32; d <<= 1) {
        unsigned int num = ShiftUp(totbits, d);
        if (tx >= d)
            totbits += num;
    }
    int pos = totbits - numbits;
    for (int y = 0; y < yloops; y++) {
        int ypos = MINMAX_H * blockIdx.y + y;
        if (ptbits & (1 << y) && pos < MEMWID) {
            points[2 * pos + 0] = xpos - 1;
            points[2 * pos + 1] = ypos;
            pos++;
        }
    }

    totbits = Shuffle(totbits, 31);
    if (tx < totbits) {
        int xpos = points[2 * tx + 0];
        int ypos = points[2 * tx + 1];
        int ptr = xpos + (ypos + (scale + 1) * height) * pitch;
        float val = d_Data0[ptr];
        float* data1 = &d_Data0[ptr];
        float dxx = 2.0f * val - data1[-1] - data1[1];
        float dyy = 2.0f * val - data1[-pitch] - data1[pitch];
        float dxy = 0.25f * (data1[+pitch + 1] + data1[-pitch - 1] - data1[-pitch + 1] - data1[+pitch - 1]);
        float tra = dxx + dyy;
        float det = dxx * dyy - dxy * dxy;
        if (tra * tra < edgeLimit * det) {
            float edge = __fdividef(tra * tra, det);
            float dx = 0.5f * (data1[1] - data1[-1]);
            float dy = 0.5f * (data1[pitch] - data1[-pitch]);
            float* data0 = d_Data0 + ptr - height * pitch;
            float* data2 = d_Data0 + ptr + height * pitch;
            float ds = 0.5f * (data0[0] - data2[0]);
            float dss = 2.0f * val - data2[0] - data0[0];
            float dxs = 0.25f * (data2[1] + data0[-1] - data0[1] - data2[-1]);
            float dys = 0.25f * (data2[pitch] + data0[-pitch] - data2[-pitch] - data0[pitch]);
            float idxx = dyy * dss - dys * dys;
            float idxy = dys * dxs - dxy * dss;
            float idxs = dxy * dys - dyy * dxs;
            float idet = __fdividef(1.0f, idxx * dxx + idxy * dxy + idxs * dxs);
            float idyy = dxx * dss - dxs * dxs;
            float idys = dxy * dxs - dxx * dys;
            float idss = dxx * dyy - dxy * dxy;
            float pdx = idet * (idxx * dx + idxy * dy + idxs * ds);
            float pdy = idet * (idxy * dx + idyy * dy + idys * ds);
            float pds = idet * (idxs * dx + idys * dy + idss * ds);
            if (pdx < -0.5f || pdx>0.5f || pdy < -0.5f || pdy>0.5f || pds < -0.5f || pds>0.5f) {
                pdx = __fdividef(dx, dxx);
                pdy = __fdividef(dy, dyy);
                pds = __fdividef(ds, dss);
            }
            float dval = 0.5f * (dx * pdx + dy * pdy + ds * pds);
            int maxPts = d_MaxNumPoints;
            float sc = powf(2.0f, (float)scale / NUM_SCALES) * exp2f(pds * factor);
            if (sc >= lowestScale) {
                atomicMax(&d_PointCounter[2 * octave + 0], d_PointCounter[2 * octave - 1]);
                unsigned int idx = atomicInc(&d_PointCounter[2 * octave + 0], 0x7fffffff);
                idx = (idx >= maxPts ? maxPts - 1 : idx);
                d_Sift[idx].xpos = xpos + pdx;
                d_Sift[idx].ypos = ypos + pdy;
                d_Sift[idx].scale = sc;
                d_Sift[idx].sharpness = val + dval;
                d_Sift[idx].edgeness = edge;
                d_Sift[idx].subsampling = subsampling;
            }
        }
    }
}

__global__ void ComputeOrientationsCONST_kernel(cudaTextureObject_t texObj, SiftPoint* d_Sift, int octave)
{
    __shared__ float hist[64];
    __shared__ float gauss[11];
    const int tx = threadIdx.x;

    int fstPts = min(d_PointCounter[2 * octave - 1], d_MaxNumPoints);
    int totPts = min(d_PointCounter[2 * octave + 0], d_MaxNumPoints);
    for (int bx = blockIdx.x + fstPts; bx < totPts; bx += gridDim.x) {

        float i2sigma2 = -1.0f / (2.0f * 1.5f * 1.5f * d_Sift[bx].scale * d_Sift[bx].scale);
        if (tx < 11)
            gauss[tx] = exp(i2sigma2 * (tx - 5) * (tx - 5));
        if (tx < 64)
            hist[tx] = 0.0f;
        __syncthreads();
        float xp = d_Sift[bx].xpos - 4.5f;
        float yp = d_Sift[bx].ypos - 4.5f;
        int yd = tx / 11;
        int xd = tx - yd * 11;
        float xf = xp + xd;
        float yf = yp + yd;
        if (yd < 11) {
            float dx = tex2D<float>(texObj, xf + 1.0, yf) - tex2D<float>(texObj, xf - 1.0, yf);
            float dy = tex2D<float>(texObj, xf, yf + 1.0) - tex2D<float>(texObj, xf, yf - 1.0);
            int bin = 16.0f * atan2f(dy, dx) / 3.1416f + 16.5f;
            if (bin > 31)
                bin = 0;
            float grad = sqrtf(dx * dx + dy * dy);
            atomicAdd(&hist[bin], grad * gauss[xd] * gauss[yd]);
        }
        __syncthreads();
        int x1m = (tx >= 1 ? tx - 1 : tx + 31);
        int x1p = (tx <= 30 ? tx + 1 : tx - 31);
        if (tx < 32) {
            int x2m = (tx >= 2 ? tx - 2 : tx + 30);
            int x2p = (tx <= 29 ? tx + 2 : tx - 30);
            hist[tx + 32] = 6.0f * hist[tx] + 4.0f * (hist[x1m] + hist[x1p]) + (hist[x2m] + hist[x2p]);
        }
        __syncthreads();
        if (tx < 32) {
            float v = hist[32 + tx];
            hist[tx] = (v > hist[32 + x1m] && v >= hist[32 + x1p] ? v : 0.0f);
        }
        __syncthreads();
        if (tx == 0) {
            float maxval1 = 0.0;
            float maxval2 = 0.0;
            int i1 = -1;
            int i2 = -1;
            for (int i = 0; i < 32; i++) {
                float v = hist[i];
                if (v > maxval1) {
                    maxval2 = maxval1;
                    maxval1 = v;
                    i2 = i1;
                    i1 = i;
                }
                else if (v > maxval2) {
                    maxval2 = v;
                    i2 = i;
                }
            }
            float val1 = hist[32 + ((i1 + 1) & 31)];
            float val2 = hist[32 + ((i1 + 31) & 31)];
            float peak = i1 + 0.5f * (val1 - val2) / (2.0f * maxval1 - val1 - val2);
            d_Sift[bx].orientation = 11.25f * (peak < 0.0f ? peak + 32.0f : peak);
            atomicMax(&d_PointCounter[2 * octave + 1], d_PointCounter[2 * octave + 0]);
            if (maxval2 > 0.8f * maxval1 && true) {
                float val1 = hist[32 + ((i2 + 1) & 31)];
                float val2 = hist[32 + ((i2 + 31) & 31)];
                float peak = i2 + 0.5f * (val1 - val2) / (2.0f * maxval2 - val1 - val2);
                unsigned int idx = atomicInc(&d_PointCounter[2 * octave + 1], 0x7fffffff);
                if (idx < d_MaxNumPoints) {
                    d_Sift[idx].xpos = d_Sift[bx].xpos;
                    d_Sift[idx].ypos = d_Sift[bx].ypos;
                    d_Sift[idx].scale = d_Sift[bx].scale;
                    d_Sift[idx].sharpness = d_Sift[bx].sharpness;
                    d_Sift[idx].edgeness = d_Sift[bx].edgeness;
                    d_Sift[idx].orientation = 11.25f * (peak < 0.0f ? peak + 32.0f : peak);;
                    d_Sift[idx].subsampling = d_Sift[bx].subsampling;
                }
            }
        }
        __syncthreads();
    }
}

__global__ void ExtractSiftDescriptorsCONSTNew_kernel(cudaTextureObject_t texObj, SiftPoint* d_sift, float subsampling, int octave)
{
    __shared__ float gauss[16];
    __shared__ float buffer[128];
    __shared__ float sums[4];

    const int tx = threadIdx.x; // 0 -> 16
    const int ty = threadIdx.y; // 0 -> 8
    const int idx = ty * 16 + tx;
    if (ty == 0)
        gauss[tx] = __expf(-(tx - 7.5f) * (tx - 7.5f) / 128.0f);

    int fstPts = min(d_PointCounter[2 * octave - 1], d_MaxNumPoints);
    int totPts = min(d_PointCounter[2 * octave + 1], d_MaxNumPoints);
    //if (tx==0 && ty==0)
    //  printf("%d %d %d %d\n", octave, fstPts, min(d_PointCounter[2*octave], d_MaxNumPoints), totPts); 
    for (int bx = blockIdx.x + fstPts; bx < totPts; bx += gridDim.x) {

        buffer[idx] = 0.0f;
        __syncthreads();

        // Compute angles and gradients
        float theta = 2.0f * 3.1415f / 360.0f * d_sift[bx].orientation;
        float sina = __sinf(theta);           // cosa -sina
        float cosa = __cosf(theta);           // sina  cosa
        float scale = 12.0f / 16.0f * d_sift[bx].scale;
        float ssina = scale * sina;
        float scosa = scale * cosa;

        for (int y = ty; y < 16; y += 8) {
            float xpos = d_sift[bx].xpos + (tx - 7.5f) * scosa - (y - 7.5f) * ssina + 0.5f;
            float ypos = d_sift[bx].ypos + (tx - 7.5f) * ssina + (y - 7.5f) * scosa + 0.5f;
            float dx = tex2D<float>(texObj, xpos + cosa, ypos + sina) -
                tex2D<float>(texObj, xpos - cosa, ypos - sina);
            float dy = tex2D<float>(texObj, xpos - sina, ypos + cosa) -
                tex2D<float>(texObj, xpos + sina, ypos - cosa);
            float grad = gauss[y] * gauss[tx] * __fsqrt_rn(dx * dx + dy * dy);
            float angf = 4.0f / 3.1415f * FastAtan2(dy, dx) + 4.0f;

            int hori = (tx + 2) / 4 - 1;      // Convert from (tx,y,angle) to bins      
            float horf = (tx - 1.5f) / 4.0f - hori;
            float ihorf = 1.0f - horf;
            int veri = (y + 2) / 4 - 1;
            float verf = (y - 1.5f) / 4.0f - veri;
            float iverf = 1.0f - verf;
            int angi = angf;
            int angp = (angi < 7 ? angi + 1 : 0);
            angf -= angi;
            float iangf = 1.0f - angf;

            int hist = 8 * (4 * veri + hori);   // Each gradient measure is interpolated 
            int p1 = angi + hist;           // in angles, xpos and ypos -> 8 stores
            int p2 = angp + hist;
            if (tx >= 2) {
                float grad1 = ihorf * grad;
                if (y >= 2) {   // Upper left
                    float grad2 = iverf * grad1;
                    atomicAdd(buffer + p1, iangf * grad2);
                    atomicAdd(buffer + p2, angf * grad2);
                }
                if (y <= 13) {  // Lower left
                    float grad2 = verf * grad1;
                    atomicAdd(buffer + p1 + 32, iangf * grad2);
                    atomicAdd(buffer + p2 + 32, angf * grad2);
                }
            }
            if (tx <= 13) {
                float grad1 = horf * grad;
                if (y >= 2) {    // Upper right
                    float grad2 = iverf * grad1;
                    atomicAdd(buffer + p1 + 8, iangf * grad2);
                    atomicAdd(buffer + p2 + 8, angf * grad2);
                }
                if (y <= 13) {   // Lower right
                    float grad2 = verf * grad1;
                    atomicAdd(buffer + p1 + 40, iangf * grad2);
                    atomicAdd(buffer + p2 + 40, angf * grad2);
                }
            }
        }
        __syncthreads();

        // Normalize twice and suppress peaks first time
        float sum = buffer[idx] * buffer[idx];
        for (int i = 16; i > 0; i /= 2)
            sum += ShiftDown(sum, i);
        if ((idx & 31) == 0)
            sums[idx / 32] = sum;
        __syncthreads();
        float tsum1 = sums[0] + sums[1] + sums[2] + sums[3];
        tsum1 = min(buffer[idx] * rsqrtf(tsum1), 0.2f);

        sum = tsum1 * tsum1;
        for (int i = 16; i > 0; i /= 2)
            sum += ShiftDown(sum, i);
        if ((idx & 31) == 0)
            sums[idx / 32] = sum;
        __syncthreads();

        float tsum2 = sums[0] + sums[1] + sums[2] + sums[3];
        float* desc = d_sift[bx].data;
        desc[idx] = tsum1 * rsqrtf(tsum2);
        if (idx == 0) {
            d_sift[bx].xpos *= subsampling;
            d_sift[bx].ypos *= subsampling;
            d_sift[bx].scale *= subsampling;
        }
        __syncthreads();
    }
}

void ExtractSift(SiftData& siftData, CudaImage& img, int numOctaves, double initBlur, float thresh, float lowestScale, bool scaleUp, float* tempMemory)
{
    unsigned int* d_PointCounterAddr;
    safeCall(cudaGetSymbolAddress((void**)&d_PointCounterAddr, d_PointCounter));
    safeCall(cudaMemset(d_PointCounterAddr, 0, (8 * 2 + 1) * sizeof(int)));
    safeCall(cudaMemcpyToSymbol(d_MaxNumPoints, &siftData.maxPts, sizeof(int)));

    const int nd = NUM_SCALES + 3;
    int w = img.width * (scaleUp ? 2 : 1);
    int h = img.height * (scaleUp ? 2 : 1);
    int p = iAlignUp(w, 128);
    int width = w, height = h;
    int size = h * p;                 // image sizes
    int sizeTmp = nd * h * p;           // laplace buffer sizes
    for (int i = 0; i < numOctaves; i++) {
        w /= 2;
        h /= 2;
        int p = iAlignUp(w, 128);
        size += h * p;
        sizeTmp += nd * h * p;
    }
    float* memoryTmp = tempMemory;
    size += sizeTmp;
    if (!tempMemory) {
        size_t pitch;
        safeCall(cudaMallocPitch((void**)&memoryTmp, &pitch, (size_t)4096, (size + 4095) / 4096 * sizeof(float)));
#ifdef VERBOSE
        printf("Allocated memory size: %d bytes\n", size);
        printf("Memory allocation time =      %.2f ms\n\n", timer.read());
#endif
    }
    float* memorySub = memoryTmp + sizeTmp;

    CudaImage lowImg;
    lowImg.Allocate(width, height, iAlignUp(width, 128), false, memorySub);
    if (!scaleUp) {
        float kernel[8 * 12 * 16];
        PrepareLaplaceKernels(numOctaves, 0.0f, kernel);
        safeCall(cudaMemcpyToSymbolAsync(d_LaplaceKernel, kernel, 8 * 12 * 16 * sizeof(float)));
        LowPass(lowImg, img, max(initBlur, 0.001f));
        ExtractSiftLoop(siftData, lowImg, numOctaves, 0.0f, thresh, lowestScale, 1.0f, memoryTmp, memorySub + height * iAlignUp(width, 128));
        safeCall(cudaMemcpy(&siftData.numPts, &d_PointCounterAddr[2 * numOctaves], sizeof(int), cudaMemcpyDeviceToHost));
        siftData.numPts = (siftData.numPts < siftData.maxPts ? siftData.numPts : siftData.maxPts);
    }

    if (!tempMemory)
        safeCall(cudaFree(memoryTmp));
#ifdef MANAGEDMEM
    safeCall(cudaDeviceSynchronize());
#else
    if (siftData.h_data)
        safeCall(cudaMemcpy(siftData.h_data, siftData.d_data, sizeof(SiftPoint) * siftData.numPts, cudaMemcpyDeviceToHost));
#endif
}

void PrepareLaplaceKernels(int numOctaves, float initBlur, float* kernel)
{
    if (numOctaves > 1) {
        float totInitBlur = (float)sqrt(initBlur * initBlur + 0.5f * 0.5f) / 2.0f;
        PrepareLaplaceKernels(numOctaves - 1, totInitBlur, kernel);
    }
    float scale = pow(2.0f, -1.0f / NUM_SCALES);
    float diffScale = pow(2.0f, 1.0f / NUM_SCALES);
    for (int i = 0; i < NUM_SCALES + 3; i++) {
        float kernelSum = 0.0f;
        float var = scale * scale - initBlur * initBlur;
        for (int j = 0; j <= LAPLACE_R; j++) {
            kernel[numOctaves * 12 * 16 + 16 * i + j] = (float)expf(-(double)j * j / 2.0 / var);
            kernelSum += (j == 0 ? 1 : 2) * kernel[numOctaves * 12 * 16 + 16 * i + j];
        }
        for (int j = 0; j <= LAPLACE_R; j++)
            kernel[numOctaves * 12 * 16 + 16 * i + j] /= kernelSum;
        scale *= diffScale;
    }
}

double LowPass(CudaImage& res, CudaImage& src, float scale)
{
    float kernel[2 * LOWPASS_R + 1];
    static float oldScale = -1.0f;
    if (scale != oldScale) {
        float kernelSum = 0.0f;
        float ivar2 = 1.0f / (2.0f * scale * scale);
        for (int j = -LOWPASS_R; j <= LOWPASS_R; j++) {
            kernel[j + LOWPASS_R] = (float)expf(-(double)j * j * ivar2);
            kernelSum += kernel[j + LOWPASS_R];
        }
        for (int j = -LOWPASS_R; j <= LOWPASS_R; j++)
            kernel[j + LOWPASS_R] /= kernelSum;
        safeCall(cudaMemcpyToSymbol(d_LowPassKernel, kernel, (2 * LOWPASS_R + 1) * sizeof(float)));
        oldScale = scale;
    }
    int width = res.width;
    int pitch = res.pitch;
    int height = res.height;
    dim3 blocks(iDivUp(width, LOWPASS_W), iDivUp(height, LOWPASS_H));
#if 1
    dim3 threads(LOWPASS_W + 2 * LOWPASS_R, 4);
    LowPassBlock_kernel << <blocks, threads >> > (src.d_data, res.d_data, width, pitch, height);
#else
    dim3 threads(LOWPASS_W + 2 * LOWPASS_R, LOWPASS_H);
    LowPass << <blocks, threads >> > (src.d_data, res.d_data, width, pitch, height);
#endif
    checkMsg("LowPass() execution failed\n");
    return 0.0;
}

int ExtractSiftLoop(SiftData& siftData, CudaImage& img, int numOctaves, double initBlur, float thresh, float lowestScale, float subsampling, float* memoryTmp, float* memorySub)
{
#ifdef VERBOSE
    TimerGPU timer(0);
#endif
    int w = img.width;
    int h = img.height;
    if (numOctaves > 1) {
        CudaImage subImg;
        int p = iAlignUp(w / 2, 128);
        subImg.Allocate(w / 2, h / 2, p, false, memorySub);
        ScaleDown(subImg, img, 0.5f);
        float totInitBlur = (float)sqrt(initBlur * initBlur + 0.5f * 0.5f) / 2.0f;
        ExtractSiftLoop(siftData, subImg, numOctaves - 1, totInitBlur, thresh, lowestScale, subsampling * 2.0f, memoryTmp, memorySub + (h / 2) * p);
    }
    ExtractSiftOctave(siftData, img, numOctaves, thresh, lowestScale, subsampling, memoryTmp);
#ifdef VERBOSE
    double totTime = timer.read();
    printf("ExtractSift time total =      %.2f ms %d\n\n", totTime, numOctaves);
#endif
    return 0;
}

void ExtractSiftOctave(SiftData& siftData, CudaImage& img, int octave, float thresh, float lowestScale, float subsampling, float* memoryTmp)
{
    const int nd = NUM_SCALES + 3;
#ifdef VERBOSE
    unsigned int* d_PointCounterAddr;
    safeCall(cudaGetSymbolAddress((void**)&d_PointCounterAddr, d_PointCounter));
    unsigned int fstPts, totPts;
    safeCall(cudaMemcpy(&fstPts, &d_PointCounterAddr[2 * octave - 1], sizeof(int), cudaMemcpyDeviceToHost));
    TimerGPU timer0;
#endif
    CudaImage diffImg[nd];
    int w = img.width;
    int h = img.height;
    int p = iAlignUp(w, 128);
    for (int i = 0; i < nd - 1; i++)
        diffImg[i].Allocate(w, h, p, false, memoryTmp + i * p * h);

    // Specify texture
    struct cudaResourceDesc resDesc;
    memset(&resDesc, 0, sizeof(resDesc));
    resDesc.resType = cudaResourceTypePitch2D;
    resDesc.res.pitch2D.devPtr = img.d_data;
    resDesc.res.pitch2D.width = img.width;
    resDesc.res.pitch2D.height = img.height;
    resDesc.res.pitch2D.pitchInBytes = img.pitch * sizeof(float);
    resDesc.res.pitch2D.desc = cudaCreateChannelDesc<float>();
    // Specify texture object parameters
    struct cudaTextureDesc texDesc;
    memset(&texDesc, 0, sizeof(texDesc));
    texDesc.addressMode[0] = cudaAddressModeClamp;
    texDesc.addressMode[1] = cudaAddressModeClamp;
    texDesc.filterMode = cudaFilterModeLinear;
    texDesc.readMode = cudaReadModeElementType;
    texDesc.normalizedCoords = 0;
    // Create texture object
    cudaTextureObject_t texObj = 0;
    cudaCreateTextureObject(&texObj, &resDesc, &texDesc, NULL);

#ifdef VERBOSE
    TimerGPU timer1;
#endif
    float baseBlur = pow(2.0f, -1.0f / NUM_SCALES);
    float diffScale = pow(2.0f, 1.0f / NUM_SCALES);
    LaplaceMulti(texObj, img, diffImg, octave);
    FindPointsMulti(diffImg, siftData, thresh, 10.0f, 1.0f / NUM_SCALES, lowestScale / subsampling, subsampling, octave);
#ifdef VERBOSE
    double gpuTimeDoG = timer1.read();
    TimerGPU timer4;
#endif
    ComputeOrientations(texObj, img, siftData, octave);
    ExtractSiftDescriptors(texObj, siftData, subsampling, octave);
    //OrientAndExtract(texObj, siftData, subsampling, octave); 

    safeCall(cudaDestroyTextureObject(texObj));
#ifdef VERBOSE
    double gpuTimeSift = timer4.read();
    double totTime = timer0.read();
    printf("GPU time : %.2f ms + %.2f ms + %.2f ms = %.2f ms\n", totTime - gpuTimeDoG - gpuTimeSift, gpuTimeDoG, gpuTimeSift, totTime);
    safeCall(cudaMemcpy(&totPts, &d_PointCounterAddr[2 * octave + 1], sizeof(int), cudaMemcpyDeviceToHost));
    totPts = (totPts < siftData.maxPts ? totPts : siftData.maxPts);
    if (totPts > 0)
        printf("           %.2f ms / DoG,  %.4f ms / Sift,  #Sift = %d\n", gpuTimeDoG / NUM_SCALES, gpuTimeSift / (totPts - fstPts), totPts - fstPts);
#endif
}

double ScaleDown(CudaImage& res, CudaImage& src, float variance)
{
    static float oldVariance = -1.0f;
    if (res.d_data == NULL || src.d_data == NULL) {
        printf("ScaleDown: missing data\n");
        return 0.0;
    }
    if (oldVariance != variance) {
        float h_Kernel[5];
        float kernelSum = 0.0f;
        for (int j = 0; j < 5; j++) {
            h_Kernel[j] = (float)expf(-(double)(j - 2) * (j - 2) / 2.0 / variance);
            kernelSum += h_Kernel[j];
        }
        for (int j = 0; j < 5; j++)
            h_Kernel[j] /= kernelSum;
        safeCall(cudaMemcpyToSymbol(d_ScaleDownKernel, h_Kernel, 5 * sizeof(float)));
        oldVariance = variance;
    }
#if 0
    dim3 blocks(iDivUp(src.width, SCALEDOWN_W), iDivUp(src.height, SCALEDOWN_H));
    dim3 threads(SCALEDOWN_W + 4, SCALEDOWN_H + 4);
    ScaleDownDenseShift << <blocks, threads >> > (res.d_data, src.d_data, src.width, src.pitch, src.height, res.pitch);
#else
    dim3 blocks(iDivUp(src.width, SCALEDOWN_W), iDivUp(src.height, SCALEDOWN_H));
    dim3 threads(SCALEDOWN_W + 4);
    ScaleDown_kernel << <blocks, threads >> > (res.d_data, src.d_data, src.width, src.pitch, src.height, res.pitch);
#endif
    checkMsg("ScaleDown() execution failed\n");
    return 0.0;
}

double LaplaceMulti(cudaTextureObject_t texObj, CudaImage& baseImage, CudaImage* results, int octave)
{
    int width = results[0].width;
    int pitch = results[0].pitch;
    int height = results[0].height;
#if 1
    dim3 threads(LAPLACE_W + 2 * LAPLACE_R);
    dim3 blocks(iDivUp(width, LAPLACE_W), height);
    LaplaceMultiMem_kernel << <blocks, threads >> > (baseImage.d_data, results[0].d_data, width, pitch, height, octave);
#endif
#if 0
    dim3 threads(LAPLACE_W + 2 * LAPLACE_R, LAPLACE_S);
    dim3 blocks(iDivUp(width, LAPLACE_W), iDivUp(height, LAPLACE_H));
    LaplaceMultiMemTest << <blocks, threads >> > (baseImage.d_data, results[0].d_data, width, pitch, height, octave);
#endif
#if 0
    dim3 threads(LAPLACE_W + 2 * LAPLACE_R, LAPLACE_S);
    dim3 blocks(iDivUp(width, LAPLACE_W), height);
    LaplaceMultiMemOld << <blocks, threads >> > (baseImage.d_data, results[0].d_data, width, pitch, height, octave);
#endif
#if 0
    dim3 threads(LAPLACE_W + 2 * LAPLACE_R, LAPLACE_S);
    dim3 blocks(iDivUp(width, LAPLACE_W), height);
    LaplaceMultiTex << <blocks, threads >> > (texObj, results[0].d_data, width, pitch, height, octave);
#endif
    checkMsg("LaplaceMulti() execution failed\n");
    return 0.0;
}

double FindPointsMulti(CudaImage* sources, SiftData& siftData, float thresh, float edgeLimit, float factor, float lowestScale, float subsampling, int octave)
{
    if (sources->d_data == NULL) {
        printf("FindPointsMulti: missing data\n");
        return 0.0;
    }
    int w = sources->width;
    int p = sources->pitch;
    int h = sources->height;
#if 0
    dim3 blocks(iDivUp(w, MINMAX_W) * NUM_SCALES, iDivUp(h, MINMAX_H));
    dim3 threads(MINMAX_W + 2, MINMAX_H);
    FindPointsMultiTest << <blocks, threads >> > (sources->d_data, siftData.d_data, w, p, h, subsampling, lowestScale, thresh, factor, edgeLimit, octave);
#endif
#if 1
    dim3 blocks(iDivUp(w, MINMAX_W) * NUM_SCALES, iDivUp(h, MINMAX_H));
    dim3 threads(MINMAX_W + 2);
#ifdef MANAGEDMEM
    FindPointsMulti << <blocks, threads >> > (sources->d_data, siftData.m_data, w, p, h, subsampling, lowestScale, thresh, factor, edgeLimit, octave);
#else
    FindPointsMultiNew_kernel << <blocks, threads >> > (sources->d_data, siftData.d_data, w, p, h, subsampling, lowestScale, thresh, factor, edgeLimit, octave);
#endif
#endif
    checkMsg("FindPointsMulti() execution failed\n");
    return 0.0;
}

double ComputeOrientations(cudaTextureObject_t texObj, CudaImage& src, SiftData& siftData, int octave)
{
    dim3 blocks(512);
#ifdef MANAGEDMEM
    ComputeOrientationsCONST << <blocks, threads >> > (texObj, siftData.m_data, octave);
#else
#if 1
    dim3 threads(11 * 11);
    ComputeOrientationsCONST_kernel << <blocks, threads >> > (texObj, siftData.d_data, octave);
#else
    dim3 threads(256);
    ComputeOrientationsCONSTNew << <blocks, threads >> > (src.d_data, src.width, src.pitch, src.height, siftData.d_data, octave);
#endif
#endif
    checkMsg("ComputeOrientations() execution failed\n");
    return 0.0;
}

double ExtractSiftDescriptors(cudaTextureObject_t texObj, SiftData& siftData, float subsampling, int octave)
{
    dim3 blocks(512);
    dim3 threads(16, 8);
#ifdef MANAGEDMEM
    ExtractSiftDescriptorsCONST << <blocks, threads >> > (texObj, siftData.m_data, subsampling, octave);
#else
    ExtractSiftDescriptorsCONSTNew_kernel << <blocks, threads >> > (texObj, siftData.d_data, subsampling, octave);
#endif
    checkMsg("ExtractSiftDescriptors() execution failed\n");
    return 0.0;
}


#endif

#ifndef MATCHUTILS
#define MATCHUTILS

__global__ void CleanMatches_kernel(SiftPoint *sift1, int numPts1)
{
  const int p1 = min(blockIdx.x*64 + threadIdx.x, numPts1-1);
  sift1[p1].score = 0.0f;
}

#define M7W   32
#define M7H   32
#define M7R    4
#define NRX    2
#define NDIM 128

__global__ void FindMaxCorr10_kernel(SiftPoint *sift1, SiftPoint *sift2, int numPts1, int numPts2)
{
  __shared__ float4 buffer1[M7W*NDIM/4]; 
  __shared__ float4 buffer2[M7H*NDIM/4];       
  int tx = threadIdx.x;
  int ty = threadIdx.y;
  int bp1 = M7W*blockIdx.x;
  for (int j=ty;j<M7W;j+=M7H/M7R) {    
    int p1 = min(bp1 + j, numPts1 - 1);
    for (int d=tx;d<NDIM/4;d+=M7W)
      buffer1[j*NDIM/4 + (d + j)%(NDIM/4)] = ((float4*)&sift1[p1].data)[d];
  }
      
  float max_score[NRX];
  float sec_score[NRX];
  int index[NRX];
  for (int i=0;i<NRX;i++) {
    max_score[i] = 0.0f;
    sec_score[i] = 0.0f;
    index[i] = -1;
  }
  int idx = ty*M7W + tx;
  int ix = idx%(M7W/NRX);
  int iy = idx/(M7W/NRX);
  for (int bp2=0;bp2<numPts2 - M7H + 1;bp2+=M7H) {
    for (int j=ty;j<M7H;j+=M7H/M7R) {      
      int p2 = min(bp2 + j, numPts2 - 1);
      for (int d=tx;d<NDIM/4;d+=M7W)
	buffer2[j*NDIM/4 + d] = ((float4*)&sift2[p2].data)[d];
    }
    __syncthreads();

    if (idx<M7W*M7H/M7R/NRX) {
      float score[M7R][NRX];                                    
      for (int dy=0;dy<M7R;dy++)
	for (int i=0;i<NRX;i++)
	  score[dy][i] = 0.0f;
      for (int d=0;d<NDIM/4;d++) {
	float4 v1[NRX];
	for (int i=0;i<NRX;i++) 
	  v1[i] = buffer1[((M7W/NRX)*i + ix)*NDIM/4 + (d + (M7W/NRX)*i + ix)%(NDIM/4)];
	for (int dy=0;dy<M7R;dy++) {
	  float4 v2 = buffer2[(M7R*iy + dy)*(NDIM/4) + d];    
	  for (int i=0;i<NRX;i++) {
	    score[dy][i] += v1[i].x*v2.x;
	    score[dy][i] += v1[i].y*v2.y;
	    score[dy][i] += v1[i].z*v2.z;
	    score[dy][i] += v1[i].w*v2.w;
	  }
	}
      }
      for (int dy=0;dy<M7R;dy++) {
	for (int i=0;i<NRX;i++) {
	  if (score[dy][i]>max_score[i]) {
	    sec_score[i] = max_score[i];
	    max_score[i] = score[dy][i];     
	    index[i] = min(bp2 + M7R*iy + dy, numPts2-1);
	  } else if (score[dy][i]>sec_score[i])
	    sec_score[i] = score[dy][i]; 
	}
      }
    }
    __syncthreads();
  }

  float *scores1 = (float*)buffer1;
  float *scores2 = &scores1[M7W*M7H/M7R];
  int *indices = (int*)&scores2[M7W*M7H/M7R];
  if (idx<M7W*M7H/M7R/NRX) {
    for (int i=0;i<NRX;i++) {
      scores1[iy*M7W + (M7W/NRX)*i + ix] = max_score[i];  
      scores2[iy*M7W + (M7W/NRX)*i + ix] = sec_score[i];  
      indices[iy*M7W + (M7W/NRX)*i + ix] = index[i];
    }
  }
  __syncthreads();
  
  if (ty==0) {
    float max_score = scores1[tx];
    float sec_score = scores2[tx];
    int index = indices[tx];
    for (int y=0;y<M7H/M7R;y++)
      if (index != indices[y*M7W + tx]) {
	if (scores1[y*M7W + tx]>max_score) {
	  sec_score = max(max_score, sec_score);
	  max_score = scores1[y*M7W + tx]; 
	  index = indices[y*M7W + tx];
	} else if (scores1[y*M7W + tx]>sec_score)
	  sec_score = scores1[y*M7W + tx];
      }
    sift1[bp1 + tx].score = max_score;
    sift1[bp1 + tx].match = index;
    sift1[bp1 + tx].match_xpos = sift2[index].xpos;
    sift1[bp1 + tx].match_ypos = sift2[index].ypos;
    sift1[bp1 + tx].ambiguity = sec_score / (max_score + 1e-6f);
  }
}


double MatchSiftData(SiftData &data1, SiftData &data2)
{
  int numPts1 = data1.numPts;
  int numPts2 = data2.numPts;
  if (!numPts1 || !numPts2) 
    return 0.0;
#ifdef MANAGEDMEM
  SiftPoint *sift1 = data1.m_data;
  SiftPoint *sift2 = data2.m_data;
#else
  if (data1.d_data==NULL || data2.d_data==NULL)
    return 0.0f;
  SiftPoint *sift1 = data1.d_data;
  SiftPoint *sift2 = data2.d_data;
#endif
  
// Original version with correlation and maximization in two different kernels
// Global memory reguirement: O(N^2)
#if 0
  float *d_corrData; 
  int corrWidth = iDivUp(numPts2, 16)*16;
  int corrSize = sizeof(float)*numPts1*corrWidth;
  safeCall(cudaMalloc((void **)&d_corrData, corrSize));
#if 0 // K40c 10.9ms, 1080 Ti 3.8ms
  dim3 blocks1(numPts1, iDivUp(numPts2, 16));
  dim3 threads1(16, 16); // each block: 1 points x 16 points
  MatchSiftPoints<<<blocks1, threads1>>>(sift1, sift2, d_corrData, numPts1, numPts2);
#else // K40c 7.6ms, 1080 Ti 1.4ms
  dim3 blocks(iDivUp(numPts1,16), iDivUp(numPts2, 16));
  dim3 threads(16, 16); // each block: 16 points x 16 points
  MatchSiftPoints2<<<blocks, threads>>>(sift1, sift2, d_corrData, numPts1, numPts2);
#endif
  safeCall(cudaDeviceSynchronize());
  dim3 blocksMax(iDivUp(numPts1, 16));
  dim3 threadsMax(16, 16);
  FindMaxCorr<<<blocksMax, threadsMax>>>(d_corrData, sift1, sift2, numPts1, corrWidth, sizeof(SiftPoint));
  safeCall(cudaDeviceSynchronize());
  checkMsg("FindMaxCorr() execution failed\n");
  safeCall(cudaFree(d_corrData));
#endif

// Version suggested by Nicholas Lin with combined correlation and maximization
// Global memory reguirement: O(N)
#if 0 // K40c 51.2ms, 1080 Ti 9.6ms
  int block_dim = 16;
  float *d_corrData;
  int corrSize = numPts1 * block_dim * 2;
  safeCall(cudaMalloc((void **)&d_corrData, sizeof(float) * corrSize));
  dim3 blocks(iDivUp(numPts1, block_dim));
  dim3 threads(block_dim, block_dim); 
  FindMaxCorr3<<<blocks, threads >>>(d_corrData, sift1, sift2, numPts1, numPts2);
  safeCall(cudaDeviceSynchronize());
  checkMsg("FindMaxCorr3() execution failed\n");
  safeCall(cudaFree(d_corrData));
#endif

// Combined version with no global memory requirement using one 1 point per block
#if 0 // K40c 8.9ms, 1080 Ti 2.1ms, 2080 Ti 1.0ms
  dim3 blocksMax(numPts1);
  dim3 threadsMax(FMC2W, FMC2H);
  FindMaxCorr2<<<blocksMax, threadsMax>>>(sift1, sift2, numPts1, numPts2);
  safeCall(cudaDeviceSynchronize());
  checkMsg("FindMaxCorr2() execution failed\n");
#endif
  
// Combined version with no global memory requirement using one FMC2H points per block
#if 0 // K40c 9.2ms, 1080 Ti 1.3ms, 2080 Ti 1.1ms
  dim3 blocksMax2(iDivUp(numPts1, FMC2H));
  dim3 threadsMax2(FMC2W, FMC2H);
  FindMaxCorr4<<<blocksMax2, threadsMax2>>>(sift1, sift2, numPts1, numPts2);
  safeCall(cudaDeviceSynchronize());
  checkMsg("FindMaxCorr4() execution failed\n");
#endif

// Combined version with no global memory requirement using global locks
#if 1
  dim3 blocksMax3(iDivUp(numPts1, 16), iDivUp(numPts2, 512));
  dim3 threadsMax3(16, 16);
  CleanMatches_kernel<<<iDivUp(numPts1, 64), 64>>>(sift1, numPts1);
  int mode = 10;
  if (mode==10) {                 // 2080 Ti 0.24ms
    blocksMax3 = dim3(iDivUp(numPts1, M7W));
    threadsMax3 = dim3(M7W, M7H/M7R);
    FindMaxCorr10_kernel<<<blocksMax3, threadsMax3>>>(sift1, sift2, numPts1, numPts2);
  }
  safeCall(cudaDeviceSynchronize());
  checkMsg("FindMaxCorr5() execution failed\n");
#endif

  if (data1.h_data!=NULL) {
    float *h_ptr = &data1.h_data[0].score;
    float *d_ptr = &data1.d_data[0].score;
    safeCall(cudaMemcpy2D(h_ptr, sizeof(SiftPoint), d_ptr, sizeof(SiftPoint), 5*sizeof(float), data1.numPts, cudaMemcpyDeviceToHost));
  }

  return 0.0;
}


#endif

void InitCuda(int devNum)
{
    int nDevices;
    cudaGetDeviceCount(&nDevices);
    if (!nDevices) {
        std::cerr << "No CUDA devices available" << std::endl;
        return;
    }
    devNum = std::min(nDevices - 1, devNum);
    deviceInit(devNum);
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, devNum);
    // printf("Device Number: %d\n", devNum);
    // printf("  Device name: %s\n", prop.name);
    // printf("  Memory Clock Rate (MHz): %d\n", prop.memoryClockRate / 1000);
    // printf("  Memory Bus Width (bits): %d\n", prop.memoryBusWidth);
    // printf("  Peak Memory Bandwidth (GB/s): %.1f\n\n",
    //     2.0 * prop.memoryClockRate * (prop.memoryBusWidth / 8) / 1.0e6);
}

int ScaleInvariantFeatureTransformCudaLauncher(float* Limg, float* Rimg, float* matched_pts, int w_, int h_, int crop_kernel_size)
{
    int w = w_;
    int h = h_;
    int k = crop_kernel_size;
    // std::cout << "Image size = (" << w << "," << h << ")" << std::endl;

    // Initial Cuda images and download images to device
    // std::cout << "Initializing data..." << std::endl;
    InitCuda(0);
    CudaImage img1, img2;
    img1.Allocate(w, h, iAlignUp(w, 128), false, NULL, Limg);
    img2.Allocate(w, h, iAlignUp(w, 128), false, NULL, Rimg);
    img1.Download();
    img2.Download();

    SiftData siftData1, siftData2;
    float initBlur = 1.0f;
    float thresh = 3.5f;
    InitSiftData(siftData1, 4096, true, true);
    InitSiftData(siftData2, 4096, true, true);
    
    float* memoryTmp = AllocSiftTempMemory(w, h, 5, false);
    //for (int i = 0; i < 1000; i++) {
    ExtractSift(siftData1, img1, 5, initBlur, thresh, 0.0f, false, memoryTmp);
    ExtractSift(siftData2, img2, 5, initBlur, thresh, 0.0f, false, memoryTmp);
    //}
    FreeSiftTempMemory(memoryTmp);

    //// Match Sift features and find a homography
    //for (int i = 0; i < 1; i++)
    MatchSiftData(siftData1, siftData2);

    SiftPoint *sift1 = siftData1.h_data;
    //SiftPoint *sift2 = siftData2.h_data;

    int numPts = siftData1.numPts;
    int cnt = 0;
    int num_matched_pts = 0;
    float ctk1 = k;
    float ctk2 = w-k-1;
    float ctk3 = k;
    float ckt4 = h-k-1;
    float ctdstw = w * 0.15;
    float ctdsth = h * 0.15;
    for (int j=0;j<numPts;j++) { 
        if (sift1[j].match_error<5) {
            float x1 = sift1[j].xpos;
            float y1 = sift1[j].ypos;
            float x2 = sift1[j].match_xpos;
            float y2 = sift1[j].match_ypos;

            if (x1 < ctk1 || x1 > ctk2 || x2 < ctk1 || x2 > ctk2 || y1 < ctk3 || y1 > ckt4 || y2 < ctk3 || y2 > ckt4) {
                continue;
            }
            if (abs(x2 - x1) > ctdstw || abs(y2 - y1) > ctdsth) {
                continue;
            }

            matched_pts[cnt+0] = x1;
            matched_pts[cnt+1] = y1;
            matched_pts[cnt+2] = x2;
            matched_pts[cnt+3] = y2;
            cnt = cnt + 4;
            num_matched_pts = num_matched_pts + 1;
        }
    }

    FreeSiftData(siftData1);
    FreeSiftData(siftData2);
    
    return num_matched_pts;
}
