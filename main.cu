#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include <sstream>
#include <random>
#include <ctime>
#include <algorithm>
#include <chrono>
#include <string>
#include <array>
#include <functional>
#include <iomanip>

template <class T, size_t S, size_t item_length>
class PinnedData
{
public:
  T* beginning {};
  T* end;
  size_t size = S;
  size_t actual_item_length;
  PinnedData(const std::string& file_name, bool bias=false):
  actual_item_length {item_length + (bias ? 1 : 0)}
  {
    cudaMallocHost((void**) &beginning, sizeof(T) * size * actual_item_length, 4);
    end = beginning;
    std::ifstream file(file_name);
    std::string data_point_string;
    if (actual_item_length == 1)
    {
      while(std::getline(file, data_point_string))
      {
        add_label(data_point_string);
      }
    }else{
      while(std::getline(file, data_point_string))
      {
        add_image(data_point_string);
      }
    }
  }
  void add_image(std::string& str)
  {
    std::stringstream line(str);
    std::string num;
    std::vector<std::string> nums;
    while(std::getline(line, num, ','))
    {
      nums.push_back(num);
    }
    for (int i = 0; i < item_length; ++i)
    {
      end[i] = static_cast<float> (std::stoi(nums[i + 1])) / 255.0;
    }
    if (actual_item_length != item_length) 
    {
      end[item_length] = 1.0f;
    }
    end += actual_item_length;
  }
  void add_label(std::string& str)
  {
    std::stringstream line(str);
    std::string num;
    std::getline(line, num, ',');
    end[0] = std::stoi(num);
    end += 1;
  }
  T* operator[](int idx)
  {
    return &beginning[idx * actual_item_length];
  }
};


float get_random_float(float min, float max)
{
    static constexpr double fraction { 1.0 / (RAND_MAX + 1.0) };  // static used for efficiency, so we only calculate this value once
    // evenly distribute the random number across our range
    return min + ((max - min) * (std::rand() * fraction));
}


void fill_with_rand(float* arr, int size, float max=0.1f)
{
  for (int i = 0; i < size; ++i)
  {
    arr[i] = get_random_float(-max, max);
  }
}

class CustomVector
{
public:
  size_t length;
  float* h_copy;
  float* d_copy{};
  CustomVector(size_t p_size, float initial_val=1):
  length{p_size}
  {
    size_t float_size = sizeof(float);
    h_copy = new float[length];
    cudaMalloc((void**) &d_copy, float_size * length);
    std::fill_n(h_copy, length, initial_val);
    cudaMemcpy(d_copy, h_copy, float_size * length, cudaMemcpyHostToDevice);
  }
  float* read()
  {
    cudaMemcpy(h_copy, d_copy, sizeof(float) * length, cudaMemcpyDeviceToHost);
    return h_copy;
  }
  friend std::ostream& operator<<(std::ostream& os, CustomVector& vec)
  {
    float* result = vec.read();
    for (int i = 0; i < vec.length; ++i)
    {
      os << result[i] << " ";
    }	
    os << '\n';
    return os;
  }
};

class CustomMatrix
{
public:
  size_t pitch;
  size_t height;
  size_t width;
  float* h_copy;
  float* d_copy{};
  CustomMatrix(size_t p_height, size_t p_width, bool const_initialization=false, float initial=0.5):
  height{p_height}, width{p_width}
  {
    size_t length = height * width;
    size_t float_size = sizeof(float);
    h_copy = new float[length];
    cudaMallocPitch((void**) &d_copy, &pitch, float_size * width, height);
    if (!const_initialization)
    {
      fill_with_rand(h_copy, length, initial);
      for (int i = 0; i < width; ++i)
      {
        h_copy[(height - 1) * width + i] = 0;
      }
    }else{
      std::fill_n(h_copy, length, initial);
    }
    cudaMemcpy2D(d_copy, pitch, h_copy, 
      width * float_size, float_size * width, height, 
      cudaMemcpyHostToDevice);
  }
  float* read()
  {
    cudaMemcpy2D(h_copy, width * sizeof(float), d_copy, 
      pitch, sizeof(float) * width, height, 
      cudaMemcpyDeviceToHost);
    return h_copy;
  }
  void write(float* content)
  {
    size_t float_size = sizeof(float);
    cudaMemcpy2D(d_copy, pitch, content, 
      width * float_size, float_size * width, height, 
      cudaMemcpyHostToDevice);	
  }
  friend std::ostream& operator<<(std::ostream& os, CustomMatrix& mat)
  {
    float* result = mat.read();
    for (int i = 0; i < mat.height; ++i)
    {
      for (int j = 0; j < mat.width; ++j)
      {
        os << result[i * mat.width + j] << " ";
      }
      os << '\n';
    }	
    return os;
  }
  void print_np()
  {
    float* result = read();
    std::cout << '[';
    for (int i = 0; i < width - 1; ++i)
    {
      std::cout << '[';
      for (int j = 0; j < height; ++j)
      {
        std::cout << result[j * width + i] << ", ";
      }
      std::cout << "], \n";
    }	
    std::cout << "] \n";

    std::cout << '[';
    for(int i = 0; i < height; ++i)
    {
      std::cout << result[i * width + width - 1] << ", ";
    }
    std::cout << "] \n";
  }
  __device__ inline float* at(int row, int col)
  {
    return (float*)((char*)d_copy + row * pitch) + col;
  }
};

class Tensor : public CustomMatrix
{
public:
  size_t height;
  size_t width;
  size_t depth;
  size_t fourth;
  float* zero {};
  Tensor(size_t p_height, size_t p_width, size_t p_depth=1, size_t p_fourth=1, 
    bool const_initialization=false, float initial=0.5):
  CustomMatrix{p_height * p_depth * p_fourth, p_width, const_initialization, initial},
  height {p_height}, width {p_width}, depth {p_depth}, fourth {p_fourth}
  {
    cudaMalloc((void **) &zero, sizeof(float));
    cudaMemset(zero, 0, sizeof(float));
  }
  friend std::ostream& operator<<(std::ostream& os, Tensor& mat)
  {
    float* result = mat.read();
    for (int f = 0; f < mat.fourth; ++f)
    {
      os << "block " << f << '\n';
      for (int d = 0; d < mat.depth; ++d)
      {
        os << "depth " << d << '\n';
        for (int i = 0; i < mat.height; ++i)
        // for (int j = 0; j < mat.width; ++j)
        {
          for (int j = 0; j < mat.width; ++j)
          // for (int i = 0; i < mat.height; ++i)
          {
            os << std::setw(10) << result[f * mat.depth * mat.height * mat.width + d * mat.height * mat.width + i * mat.width + j] << ", ";
          }
          os << '\n';
        }	
      }
    }
    return os;
  }
  void print_np_page(int depth_d, int block_d)
  {
    float* result = read();
    std::cout << '[';
    for (int i = 0; i < height; ++i)
    {
      std::cout << '[';
      for (int j = 0; j < width; ++j)
      {
        std::cout << result[block_d * depth * height * width + depth_d * height * width + i * width + j];
        if (j != width - 1) {
          std::cout << ", ";
        }
      }
      std::cout << "]";
      if (i != height - 1) {
        std::cout << ", ";
        std::cout << '\n';
      }
    }	
    std::cout << "] \n";
    
  }
  void make_file(std::string path)
  {
    float* result = read();
    std::ofstream tensor_file(path);
    tensor_file << fourth << ' ';
    tensor_file << depth << ' ';
    tensor_file << height << ' ';
    tensor_file << width << ' ';
    for (int counter = 0; counter < height * width * depth * fourth; ++counter)
    {
      tensor_file << result[counter] << ' ';
    }
    tensor_file.close();
  }
  __device__ __forceinline__ float* at(int row, int col, int page=0, int block=0)
  {
    if (row < 0 || row >= height || col < 0 || col >= width)
    {
      return zero;
    }
    return (float*)((char*)d_copy + (block * height * depth + page * height + row) * pitch) + col;
  }

  __device__ __forceinline__ float* where(int idx, int block)
  {
    return (float*)((char*)d_copy + (idx / width + block * height * depth) * pitch) + (idx % width);
  }

  __device__ __forceinline__ float* at_flipped(int row, int col, int page=0, int block=0)
  {
    if (row < 0 || row > height || col < 0 || col > width)
    {
      return zero;
    }
    return (float*)((char*)d_copy + (block * height * depth + page * height + (height - row - 1)) * pitch) + (width - col - 1);
  }
};

__global__ void matmulvec(float* mat, float* vec, int height, int width, float* out)
{
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < height)
  {
    float result = 0.0f;
    for (int j = 0; j < width; ++j)
    {
      result += mat[i * width + j] * vec[j];
    }
    out[i] = result;
  }
}

__global__ void matmulmat(Tensor left, Tensor right, Tensor out)
{
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  int j = blockIdx.y * blockDim.y + threadIdx.y;
  if (i < left.height && j < right.width)
  {
    float result = 0.0f;
    for (int loopIdx = 0; loopIdx < left.width; ++loopIdx)
    {
      result += *left.at(i, loopIdx) * (*right.at(loopIdx, j));
    }
    *out.at(i, j) = result;
  }
}

__global__ void matmulmatT(Tensor left, Tensor right, Tensor out)
{
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  int j = blockIdx.y * blockDim.y + threadIdx.y;
  if (i < left.height && j < right.height)
  {
    float result = 0.0f;
    for (int loopIdx = 0; loopIdx < left.width; ++loopIdx)
    {
      result += *left.at(i, loopIdx) * (*right.at(j, loopIdx));
    }
    *out.at(i, j) = result;
  }
}

__global__ void matmulmatTtoconv(Tensor left, Tensor right, Tensor out)
{
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  int j = blockIdx.y * blockDim.y + threadIdx.y;
  if (i < left.height && j < right.height)
  {
    float result = 0.0f;
    for (int loopIdx = 0; loopIdx < left.width; ++loopIdx)
    {
      result += *left.at(i, loopIdx) * (*right.at(j, loopIdx));
    }
    *out.where(j, i) = result;
  }
}

__global__ void convmulfc(Tensor acts, Tensor weights, Tensor out)
{
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  int j = blockIdx.y * blockDim.y + threadIdx.y;
  if (i < acts.fourth && j < weights.width)
  {
    float result = 0.0f;
    for (int loopIdx = 0; loopIdx < acts.width * acts.height * acts.depth; ++loopIdx)
    {
      result += *acts.where(loopIdx, i) * (*weights.at(loopIdx, j));
      // result += *weights.at(loopIdx, j);
      // result += 0;
    }
    *out.at(i, j) = result;
  }
}

__global__ void simple_bias(Tensor in, Tensor biases, Tensor out)
{
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  int j = blockIdx.y * blockDim.y + threadIdx.y;
  *out.at(i, j) = *in.at(i, j) + *biases.at(0, j);
}

__global__ void filter_transform(Tensor in, Tensor t_mat, Tensor out) 
{
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  int j = blockIdx.y * blockDim.y + threadIdx.y;
  int k = blockIdx.z * blockDim.z + threadIdx.z;

  __shared__ float intermediate[4][3][3][3]; //probably should be parametrized in the future (can't just put in.*** here though so ...)

  if (i < t_mat.height && j < in.width)
  {
    float result = 0.0f;
    for (int loopIdx = 0; loopIdx < t_mat.width; ++loopIdx)
    {
      result += *t_mat.at(i, loopIdx) * (*in.at(loopIdx, j, k % in.depth, k / in.depth));
    }
    // *out.at(i, j, k % in.depth, k / in.depth) = result;
    intermediate[i][j][k % in.depth][k / in.depth] = result;
  }

  if (i < t_mat.height && j < t_mat.height)
  {
    float result = 0.0f;
    for (int loopIdx = 0; loopIdx < in.width; ++loopIdx)
    {
      result += intermediate[i][loopIdx][k % in.depth][k / in.depth] * (*t_mat.at(j, loopIdx));
    }
    *out.at(i, j, k % in.depth, k / in.depth) = result;
  }
}

__global__ void flipped_filter_transform(Tensor in, Tensor t_mat, Tensor out) 
{
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  int j = blockIdx.y * blockDim.y + threadIdx.y;
  int k = blockIdx.z * blockDim.z + threadIdx.z;

  __shared__ float intermediate[4][3][3][3]; //probably should be parametrized in the future (can't just put in.*** here though so ...)

  if (i < t_mat.height && j < in.width)
  {
    float result = 0.0f;
    for (int loopIdx = 0; loopIdx < t_mat.width; ++loopIdx)
    {
      result += *t_mat.at(i, loopIdx) * (*in.at_flipped(loopIdx, j, k % in.depth, k / in.depth));
    }
    // *out.at(i, j, k % in.depth, k / in.depth) = result;
    intermediate[i][j][k % in.depth][k / in.depth] = result;
  }

  if (i < t_mat.height && j < t_mat.height)
  {
    float result = 0.0f;
    for (int loopIdx = 0; loopIdx < in.width; ++loopIdx)
    {
      result += intermediate[i][loopIdx][k % in.depth][k / in.depth] * (*t_mat.at(j, loopIdx));
    }
    *out.at(i, j, k % in.depth, k / in.depth) = result;
  }
}

__global__ void map_transform(Tensor in, Tensor t_mat, Tensor out) 
{
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  int j = blockIdx.y * blockDim.y + threadIdx.y;
  int k = blockIdx.z * blockDim.z + threadIdx.z;

  int xIdx = i * 2;
  int yIdx = j * 2;
  // int zIdx = k * 2;
  // MAGIC NUMBERS

  float intermediate[4][4];

  float result = 0.0f;

  for (int hIdx = 0; hIdx < 4; ++hIdx)
  {
    for (int vIdx = 0; vIdx < 4; ++vIdx)
    {
      result = 0;
      for (int loopIdx = 0; loopIdx < 4; ++loopIdx)
      {
        result += (*t_mat.at(vIdx, loopIdx)) * (*in.at(yIdx + loopIdx, xIdx + hIdx, k % in.depth, k / in.depth));
      }
      intermediate[vIdx][hIdx] = result;
    }
  }

  for (int hIdx = 0; hIdx < 4; ++hIdx)
  {
    for (int vIdx = 0; vIdx < 4; ++vIdx)
    {
      result = 0;
      for (int loopIdx = 0; loopIdx < 4; ++loopIdx)
      {
        result += intermediate[vIdx][loopIdx] * (*t_mat.at(hIdx, loopIdx));
      }
      *out.at((2 * yIdx + vIdx), (2 * xIdx + hIdx), (k % in.depth), (k / in.depth)) = result;
    }
  }	
}

__global__ void map_transform_for_backprop(Tensor in, Tensor t_mat, Tensor out) 
{
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  int j = blockIdx.y * blockDim.y + threadIdx.y;
  int k = blockIdx.z * blockDim.z + threadIdx.z;

  int xIdx = i * 2;
  int yIdx = j * 2;
  // int zIdx = k * 2;
  // MAGIC NUMBERS

  float intermediate[4][2];

  float result = 0.0f;

  for (int hIdx = 0; hIdx < 2; ++hIdx)
  {
    for (int vIdx = 0; vIdx < 4; ++vIdx)
    {
      result = 0;
      for (int loopIdx = 0; loopIdx < 2; ++loopIdx)
      {
        result += (*t_mat.at(vIdx, loopIdx)) * (*in.at(yIdx + loopIdx, xIdx + hIdx, k % in.depth, k / in.depth));
      }
      // *inter.at((2 * yIdx + vIdx), (2 * xIdx + hIdx), (k % in.depth), (k / in.depth)) = result;
      intermediate[vIdx][hIdx] = result;
    }
  }

  for (int hIdx = 0; hIdx < 4; ++hIdx)
  {
    for (int vIdx = 0; vIdx < 4; ++vIdx)
    {
      result = 0;
      for (int loopIdx = 0; loopIdx < 2; ++loopIdx)
      {
        // result += (*inter.at(2 * yIdx + vIdx, 2 * xIdx + loopIdx)) * (*t_mat.at(hIdx, loopIdx));
        result += intermediate[vIdx][loopIdx] * (*t_mat.at(hIdx, loopIdx));
      }
      *out.at((2 * yIdx + vIdx), (2 * xIdx + hIdx), (k % in.depth), (k / in.depth)) = result;
    }
  } 
}

__global__ void inverse_transform_for_weights(Tensor in, Tensor t_mat, float learning_coefficient, Tensor out) 
{
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  int j = blockIdx.y * blockDim.y + threadIdx.y;
  int k = blockIdx.z * blockDim.z + threadIdx.z;

  int xIdx = i * 4;
  int yIdx = j * 4;
  // int zIdx = k * 2;
  // MAGIC NUMBERS

  float intermediate[3][4];

  float result = 0.0f;

  for (int hIdx = 0; hIdx < 4; ++hIdx)
  {
    for (int vIdx = 0; vIdx < 3; ++vIdx)
    {
      result = 0;
      for (int loopIdx = 0; loopIdx < 4; ++loopIdx)
      {
        result += (*t_mat.at(vIdx, loopIdx)) * (*in.at(yIdx + loopIdx, xIdx + hIdx, k % in.depth, k / in.depth));
      }
      // *inter.at((2 * yIdx + vIdx), (2 * xIdx + hIdx), (k % in.depth), (k / in.depth)) = result;
      intermediate[vIdx][hIdx] = result;
    }
  }

  for (int hIdx = 0; hIdx < 3; ++hIdx)
  {
    for (int vIdx = 0; vIdx < 3; ++vIdx)
    {
      result = 0;
      for (int loopIdx = 0; loopIdx < 4; ++loopIdx)
      {
        result += intermediate[vIdx][loopIdx] * (*t_mat.at(hIdx, loopIdx));
      }
      *out.at((yIdx / 2 + vIdx), (xIdx / 2 + hIdx), (k % in.depth), (k / in.depth)) -= result * learning_coefficient;
      // *out.at((yIdx / 2 + vIdx), (xIdx / 2 + hIdx), (k % in.depth), (k / in.depth)) = result * learning_coefficient;
    }
  }	
}

__global__ void inverse_transform_with_bias(Tensor in, Tensor t_mat, CustomMatrix biases, Tensor out) 
{
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  int j = blockIdx.y * blockDim.y + threadIdx.y;
  int k = blockIdx.z * blockDim.z + threadIdx.z;

  int xIdx = i * 4;
  int yIdx = j * 4;
  // int zIdx = k * 2;
  // MAGIC NUMBERS

  float intermediate[2][4];

  float result = 0.0f;

  for (int hIdx = 0; hIdx < 4; ++hIdx)
  {
    for (int vIdx = 0; vIdx < 2; ++vIdx)
    {
      result = 0;
      for (int loopIdx = 0; loopIdx < 4; ++loopIdx)
      {
        result += (*t_mat.at(vIdx, loopIdx)) * (*in.at(yIdx + loopIdx, xIdx + hIdx, k % in.depth, k / in.depth));
      }
      // *inter.at((2 * yIdx + vIdx), (2 * xIdx + hIdx), (k % in.depth), (k / in.depth)) = result;
      intermediate[vIdx][hIdx] = result;
    }
  }

  for (int hIdx = 0; hIdx < 2; ++hIdx)
  {
    for (int vIdx = 0; vIdx < 2; ++vIdx)
    {
      result = 0;
      for (int loopIdx = 0; loopIdx < 4; ++loopIdx)
      {
        // result += (*inter.at(2 * yIdx + vIdx, 2 * xIdx + loopIdx)) * (*t_mat.at(hIdx, loopIdx));
        result += intermediate[vIdx][loopIdx] * (*t_mat.at(hIdx, loopIdx));
      }
      *out.at((yIdx / 2 + vIdx), (xIdx / 2 + hIdx), (k % in.depth), (k / in.depth)) = result + *biases.at(0, k % in.depth);
    }
  }	
}

__global__ void inverse_transform(Tensor in, Tensor t_mat, Tensor out) 
{
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  int j = blockIdx.y * blockDim.y + threadIdx.y;
  int k = blockIdx.z * blockDim.z + threadIdx.z;

  int xIdx = i * 4;
  int yIdx = j * 4;
  // int zIdx = k * 2;
  // MAGIC NUMBERS

  float intermediate[2][4];

  float result = 0.0f;

  for (int hIdx = 0; hIdx < 4; ++hIdx)
  {
    for (int vIdx = 0; vIdx < 2; ++vIdx)
    {
      result = 0;
      for (int loopIdx = 0; loopIdx < 4; ++loopIdx)
      {
        result += (*t_mat.at(vIdx, loopIdx)) * (*in.at(yIdx + loopIdx, xIdx + hIdx, k % in.depth, k / in.depth));
      }
      // *inter.at((2 * yIdx + vIdx), (2 * xIdx + hIdx), (k % in.depth), (k / in.depth)) = result;
      intermediate[vIdx][hIdx] = result;
    }
  }

  for (int hIdx = 0; hIdx < 2; ++hIdx)
  {
    for (int vIdx = 0; vIdx < 2; ++vIdx)
    {
      result = 0;
      for (int loopIdx = 0; loopIdx < 4; ++loopIdx)
      {
        // result += (*inter.at(2 * yIdx + vIdx, 2 * xIdx + loopIdx)) * (*t_mat.at(hIdx, loopIdx));
        result += intermediate[vIdx][loopIdx] * (*t_mat.at(hIdx, loopIdx));
      }
      *out.at((yIdx / 2 + vIdx), (xIdx / 2 + hIdx), (k % in.depth), (k / in.depth)) = result;
    }
  }	
}

__global__ void conv_relu_kernel(Tensor in, Tensor out)
{
  int xIdx = (blockIdx.x * blockDim.x + threadIdx.x) * 2;
  int yIdx = (blockIdx.y * blockDim.y + threadIdx.y) * 2;
  int k = blockIdx.z * blockDim.z + threadIdx.z;

  int dim = 2;

  for (int vIdx = 0; vIdx < dim && vIdx < in.width - yIdx; ++vIdx)
  {
    for (int hIdx = 0; hIdx < dim && hIdx < in.height - xIdx; ++hIdx)
    {
      *out.at(yIdx + vIdx, xIdx + hIdx, k % in.depth, k / in.depth) = 
        (*in.at(yIdx + vIdx, xIdx + hIdx, k % in.depth, k / in.depth) < 0) ? 
          0 : 
          *in.at(yIdx + vIdx, xIdx + hIdx, k % in.depth, k / in.depth);
    }
  }
}

__global__ void conv_relu_derivative(Tensor in, Tensor out)
{
  int xIdx = (blockIdx.x * blockDim.x + threadIdx.x) * 2;
  int yIdx = (blockIdx.y * blockDim.y + threadIdx.y) * 2;
  int k = blockIdx.z * blockDim.z + threadIdx.z;

  int dim = 2;

  for (int vIdx = 0; vIdx < dim; ++vIdx)
  {
    for (int hIdx = 0; hIdx < dim; ++hIdx)
    {
      *out.at(yIdx + vIdx, xIdx + hIdx, k % in.depth, k / in.depth) = 
        (*in.at(yIdx + vIdx, xIdx + hIdx, k % in.depth, k / in.depth) < 0.0f) ? 
          0.0f : 
          1.0f;
    }
  }
}

__global__ void wts_input_mul_filter(Tensor map, Tensor filter, Tensor out) // wts = winograd transform space
{
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  int j = blockIdx.y * blockDim.y + threadIdx.y;
  int k = blockIdx.z * blockDim.z + threadIdx.z;

  int xIdx = i * 4;
  int yIdx = j * 4;
  int mapIdx = k / filter.fourth;
  int filterIdx = k % filter.fourth;

  int alpha = filter.height;

  for (int vIdx = 0; vIdx < alpha; ++vIdx)
  {
    for (int hIdx = 0; hIdx < alpha; ++hIdx)
    {
      float result = 0;
      for (int loopIdx = 0; loopIdx < filter.depth; ++loopIdx)
      {
        result += *map.at(yIdx + vIdx, xIdx + hIdx, loopIdx, mapIdx) * (*filter.at(vIdx, hIdx, loopIdx, filterIdx));
        // result += *map.at(yIdx + vIdx, xIdx + hIdx, loopIdx, 0) * (*filter.at(vIdx, hIdx, loopIdx, 0));
      }
      *out.at(yIdx + vIdx, xIdx + hIdx, filterIdx, mapIdx) = result;
    }
  }	
}

__global__ void wts_ll_acts_mul_errs(Tensor map, Tensor filter, Tensor out)
{
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  int j = blockIdx.y * blockDim.y + threadIdx.y;
  int k = blockIdx.z * blockDim.z + threadIdx.z;

  int xIdx = i * 4;
  int yIdx = j * 4;
  int mapIdx = (k / filter.depth) / map.fourth;
  int batchIdx = (k / filter.depth) % map.fourth;
  int filterIdx = k % (filter.depth);

  for (int vIdx = 0; vIdx < 4; ++vIdx)
  {
    for (int hIdx = 0; hIdx < 4; ++hIdx)
    {
      atomicAdd(
        out.at(vIdx, hIdx, mapIdx, filterIdx),
        *map.at(yIdx + vIdx, xIdx + hIdx, mapIdx, batchIdx) * (*filter.at(yIdx + vIdx, xIdx + hIdx, filterIdx, batchIdx))
      );
      // atomicAdd(
      //   out.at(vIdx, hIdx, 0, 0),
      //   *map.at(yIdx + vIdx, xIdx + hIdx, 0, 0) * (*filter.at(yIdx + vIdx, xIdx + hIdx, 0, 0))
      // );
      // atomicAdd(
      //   out.at(vIdx, hIdx, mapIdx, filterIdx),
      //   4
      // );
      // if (filterIdx > *out.at(vIdx, hIdx, mapIdx, filterIdx))
        // *out.at(vIdx, hIdx, mapIdx, filterIdx) = k;
    }
  }
}

__global__ void wts_nle_mul_nlw(Tensor map, Tensor filter, Tensor out) // wts = winograd transform space
{
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  int j = blockIdx.y * blockDim.y + threadIdx.y;
  int k = blockIdx.z * blockDim.z + threadIdx.z;

  int xIdx = i * 4;
  int yIdx = j * 4;
  int batchIdx = k / filter.depth;
  int filterDepthIdx = k % filter.depth;

  int alpha = filter.height;

  for (int vIdx = 0; vIdx < alpha; ++vIdx)
  {
    for (int hIdx = 0; hIdx < alpha; ++hIdx)
    {
      float result = 0;
      for (int loopIdx = 0; loopIdx < filter.fourth; ++loopIdx)
      {
        result += *map.at(yIdx + vIdx, xIdx + hIdx, loopIdx, batchIdx) * (*filter.at(vIdx, hIdx, filterDepthIdx, loopIdx));
        // result += *map.at(yIdx + vIdx, xIdx + hIdx, loopIdx, 0) * (*filter.at(vIdx, hIdx, loopIdx, 0));
      }
      *out.at(yIdx + vIdx, xIdx + hIdx, filterDepthIdx, batchIdx) = result;
    }
  }	
}

__global__ void relu_kernel(Tensor in, Tensor out)
{
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  int j = blockIdx.y * blockDim.y + threadIdx.y;

  if (i < in.height && j < in.width)
    *out.at(i, j) = (*in.at(i, j) < 0) ? 0 : *in.at(i, j);
}

__global__ void sigmoid_kernel(float* input, float* output, size_t size)
{
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < size)
  {
    output[i] = 1.0f/(1.0f + expf(-input[i]));
  }
}

__global__ void sigmoid_derivative(float* input, float* output, size_t size)
{
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < size)
  {
    output[i] = 1.0f/(1.0f + expf(-input[i])) * (1 - 1.0f/(1.0f + expf(-input[i])));
  }
}

__global__ void elementwisemul(Tensor left, Tensor right, Tensor out)
{
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  int j = blockIdx.y * blockDim.y + threadIdx.y;
  int k = blockIdx.z * blockDim.z + threadIdx.z;

  int depthIdx = k / left.fourth;
  int fourthIdx = k % left.fourth;

  if (i < left.height && j < left.width)
    *out.at(i, j, depthIdx, fourthIdx) = *left.at(i, j, depthIdx, fourthIdx) * (*right.at(i, j, depthIdx, fourthIdx));
}

__global__ void relu_derivative(Tensor in, Tensor out)
{
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  int j = blockIdx.y * blockDim.y + threadIdx.y;
  if (i < in.height && j < in.width)
    *out.at(i, j) = (*in.at(i, j) < 0.0f) ? 0.0f : 1.0f;
}

__global__ void softmax_kernel(Tensor in, Tensor out)
{
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  int j = blockIdx.y * blockDim.y + threadIdx.y;
  if (i < in.height && j < in.width)
  {
    float sum = 0;
    for (int loopIdx = 0; loopIdx < in.width; ++loopIdx)
    {
      sum += expf(*in.at(i, loopIdx));
    }
    *out.at(i, j) = expf(*in.at(i, j)) / sum;
  }
}

__global__ void softmax_crossen_error(Tensor in, Tensor out, int* targets)
{
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  int j = blockIdx.y * blockDim.y + threadIdx.y;
  if (i < in.height && j < in.width)
  {
    if (j == targets[i])
    {
      *out.at(i, j) = *in.at(i, j) - 1;
    }else{
      *out.at(i, j) = *in.at(i, j);
    }
  }
}

__global__ void sigmoid_square_error(Tensor in, Tensor out, int* targets)
{
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  int j = blockIdx.y * blockDim.y + threadIdx.y;
  if (i < in.height && j < in.width)
  {
    if (i == targets[j])
    {
      *out.at(i, j) = (*in.at(i, j) - 1) * (*in.at(i, j)) * (1 - *in.at(i, j));
    }else{
      *out.at(i, j) = *in.at(i, j) * (*in.at(i, j)) * (1 - *in.at(i, j));
    }
  }
}

__global__ void cross_entropy(float* input, float* output, size_t size)
{
  // int i = blockIdx.x * blockDim.x + threadIdx.x;
}

__global__ void mean_square_error(float* input, float* output, size_t size)
{
  // int i = blockIdx.x * blockDim.x + threadIdx.x;
}

__global__ void weight_update_kernel(Tensor errors, Tensor last_activations, Tensor weights, float learning_rate)
{
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  int j = blockIdx.y * blockDim.y + threadIdx.y;
  if (i < weights.height && j < weights.width)
  {
    float result = 0.0f;
    for (int loopIdx = 0; loopIdx < errors.height; ++loopIdx)
    {
      result += *last_activations.at(loopIdx, i) * (*errors.at(loopIdx, j));
    }
    *weights.at(i, j) -= learning_rate * result * (1 / (float)errors.height);
  }
}

__global__ void weight_update_kernel_from_conv(Tensor errors, Tensor last_activations, Tensor weights, float learning_rate)
{
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  int j = blockIdx.y * blockDim.y + threadIdx.y;
  if (i < weights.height && j < weights.width)
  {
    float result = 0.0f;
    for (int loopIdx = 0; loopIdx < errors.height; ++loopIdx)
    {
      result += *last_activations.where(i, loopIdx) * (*errors.at(loopIdx, j));
    }
    *weights.at(i, j) -= learning_rate * result * (1 / (float)errors.height);
  }
}
__global__ void weight_update_kernel_for_conv(Tensor errors, Tensor last_activations, Tensor weights, float learning_rate)
{
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  int j = blockIdx.y * blockDim.y + threadIdx.y;
  if (i < 3 && j < 3)
  {
    float result = 0.0f;
    for (int loopIdx = 0; loopIdx < errors.height; ++loopIdx)
    {
      result += *last_activations.where(0, 0) * (*errors.at(0, 0));
    }
    // *weights.at(i, j) -= learning_rate * result * (1 / (float)errors.height);
    *weights.where(i, j) = 0;
  }
}

__global__ void update_correct_labels(Tensor acts, int* labels, int* correct_predictions)
{
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  int maxIdx = 0;
  for (int loopIdx = 1; loopIdx < acts.width - 1; ++loopIdx)
  {
    if (*acts.at(i, loopIdx) > *acts.at(i, maxIdx))
      maxIdx = loopIdx;
  }
  if (maxIdx == labels[i])//maxIdx == labels[i]
    atomicAdd(correct_predictions, 1);
}

__global__ void testWhere(Tensor in, size_t index, size_t block)
{
  *in.where(index, block) = 666;
}

class Activation
{
public:
  void (*f)(Tensor, Tensor);
  void (*d)(Tensor, Tensor);
  Activation(void (*p_f)(Tensor, Tensor), void (*p_d)(Tensor, Tensor)):
  f{p_f}, d{p_d}
  {} 
};

Activation relu(relu_kernel, relu_derivative);
Activation softmax(softmax_kernel, relu_derivative);
// activation sigmoid(sigmoid_kernel, sigmoid_derivative);

dim3 get_grids(size_t x_dim, size_t y_dim, size_t z_dim=1)
{
  return dim3((x_dim > 40) ? x_dim / 20 + 1 : 2, (y_dim > 40) ? y_dim / 20 + 1 : 2, z_dim);
}


dim3 get_threads(size_t x_dim, size_t y_dim)
{
  return dim3(
    x_dim / ((x_dim > 40) ? x_dim / 20 + 1 : 2) + 1, 
    y_dim / ((y_dim > 40) ? y_dim / 20 + 1 : 2) + 1);
}

class Layer
{
public:
  Tensor activations {1, 1};
  Tensor activations_alt {1, 1};
  Tensor pre_activations {1, 1};
  Tensor errors {1, 1};
  Tensor weights {1, 1};
  Activation act;
  Layer(Activation act_p):
  act {act_p}
  {}
  virtual void forward(Tensor& input, cudaStream_t s) = 0;
  virtual void backward(Tensor& nlw, Tensor& nle, cudaStream_t s) = 0;
  virtual void update_weights(std::vector<std::reference_wrapper<Layer>>::iterator ll_iterator, float learning_rate, cudaStream_t stream, bool use_alt=false) = 0;
  virtual void set_input_props(const Layer& lla) = 0;
  virtual void initialize_with_batch_size(size_t batch_size, const Layer& ll) = 0;
  virtual void initialize_with_next_layer(const Layer& nl) = 0;
  virtual size_t get_output_size() const = 0;
  virtual size_t get_output_bias_size() const = 0;
  virtual size_t get_depth() const = 0;
  virtual size_t get_height() const = 0;
  virtual size_t get_width() const = 0;
};


class Regular : public Layer
{
public:
  size_t units;
  size_t input_length;
  bool double_activations;
  Regular(size_t p_units=16, Activation act_p=relu, bool p_double_activations=false, size_t p_input_length=1):
  Layer{act_p}, units{p_units}, input_length{p_input_length}, double_activations{p_double_activations}
  {}
  void forward(Tensor& input, cudaStream_t s)
  {
    matmulmat<<<
      get_grids(input.height, units),
      get_threads(input.height, units),
      0, 
      s
    >>>(input, weights, pre_activations);
    act.f<<<
      get_grids(input.height, units),
      get_threads(input.height, units), 
      0, 
      s
    >>>(pre_activations, activations);
    cudaDeviceSynchronize();
    // std::cout << activations << '\n';
  }
  void backward(Tensor& nlw, Tensor& nle, cudaStream_t s)
  {
    matmulmatT<<<
      get_grids(nle.height, units), 
      get_threads(nle.height, units), 
      0, 
      s
    >>>(nle, nlw, errors);
    act.d<<<
      get_grids(pre_activations.height, units), 
      get_threads(pre_activations.height, units), 
      0, 
      s
    >>>(pre_activations, pre_activations);
    elementwisemul<<<
      get_grids(errors.height, units), 
      get_threads(errors.height, units), 
      0, 
      s
    >>>(errors, pre_activations, errors);
  }

  void update_weights(std::vector<std::reference_wrapper<Layer>>::iterator ll_iterator, float learning_rate, cudaStream_t stream, bool use_alt=false)
  {
    weight_update_kernel<<<
      get_grids(weights.height, weights.width),
      get_threads(weights.height, weights.width),
      0, 
      stream
    >>>(errors, (ll_iterator)->get().activations, weights, learning_rate);
  }

  void set_input_props(const Layer& ll)
  {
    input_length = ll.get_output_size() + ll.get_output_bias_size();
    // std::cout << input_length << '\n';
    weights = Tensor(input_length, units);
  }
  void initialize_with_batch_size(size_t batch_size, const Layer& ll)
  {
    activations = Tensor(batch_size, units + 1, 1, 1, true, 1.0f);
    pre_activations = Tensor(batch_size, units, 1, 1, true, 1.0f);
    errors = Tensor(batch_size, units, 1, 1, true, 1.0f);
    if (double_activations)
      activations_alt = Tensor(batch_size, units + 1, 1, 1, true, 1.0f);
  }
  void initialize_with_next_layer(const Layer& nl)
  {
    // std::cout << nl.weights.height << ' ' << nl.weights.width << '\n'; 
  }
  size_t get_output_size() const
  {
    return units;
  }
  size_t get_output_bias_size() const 
  {
    return 1;
  }
  size_t get_depth() const
  {
    return 1;
  }
  size_t get_height() const
  {
    return activations.height;
  }
  size_t get_width() const
  {
    return activations.width;
  }
};

class FCfromConv : public Regular
{
public:
  Tensor biases {1, 1};
  FCfromConv(size_t p_units=16, Activation act_p=relu, bool p_double_activations=false, size_t p_input_length=1):
  Regular(p_units, act_p, p_double_activations, p_input_length)
  {}
  void forward(Tensor& input, cudaStream_t s)
  {
    convmulfc<<<
      get_grids(input.fourth, units),
      get_threads(input.fourth, units),
      0, 
   		s
    >>>(input, weights, pre_activations);
    // simple_bias<<<
    //   get_grids(activations.height, activations.width),
    //   get_threads(activations.height, activations.width),
    //   0,
   	// 	s
    // >>>(pre_activations, biases, pre_activations);
    act.f<<<
      get_grids(input.fourth, units),
      get_threads(input.fourth, units), 
      0, 
   		s
    >>>(pre_activations, activations);
  }

  void update_weights(std::vector<std::reference_wrapper<Layer>>::iterator ll_iterator, float learning_rate, cudaStream_t stream, bool use_alt=false)
  {
    weight_update_kernel_from_conv<<<
      get_grids(weights.height, weights.width),
      get_threads(weights.height, weights.width),
      0, 
      stream
    >>>(errors, (ll_iterator)->get().activations, weights, learning_rate);
  }

  void initialize_with_batch_size(size_t batch_size, const Layer& ll)
  {
    activations = Tensor(batch_size, units + 1, 1, 1, true, 1.0f);
    biases = Tensor(1, units);
    pre_activations = Tensor(batch_size, units, 1, 1, true, 1.0f);
    errors = Tensor(batch_size, units, 1, 1, true, 1.0f);
    if (double_activations)
      activations_alt = Tensor(batch_size, units + 1, 1, 1, true, 1.0f);
    set_input_props(ll);
  }
};

class Convolutional : public Layer
{
public:
  size_t filter_quantity;
  std::array<size_t, 2> filter_dims;
  std::array<size_t, 2> map_dims;
  Tensor transformed_weights {1, 1};
  Tensor transformed_input {1, 1};
  Tensor forward_inter {1, 1};
  Tensor transformed_flipped_weights {1, 1};
  Tensor transfromed_nle {1, 1};
  Tensor backward_conv_inter {1, 1};
  Tensor ll_transformed_acts {1, 1};
  Tensor transformed_errors {1, 1};
  Tensor weight_update_inter {1, 1};
  CustomMatrix biases {1, 1};
  Tensor G_matrix {4, 3};
  Tensor G2_matrix {4, 2}; // G matrix for f(3*3, 2*2) used in backprop
  Tensor B_matrix {4, 4};
  Tensor B2_matrix {4, 4};
  Tensor A_matrix {2, 4};
  Tensor A2_matrix {3, 4};
  bool same_padding {true};
  Convolutional(size_t p_filter_quantity, std::array<size_t, 2> p_filter_dims,
    Activation act_p=relu, bool p_same_padding=true):
  Layer{act_p}, filter_quantity {p_filter_quantity}, same_padding {p_same_padding},
  filter_dims {p_filter_dims}, map_dims {0, 0}
  {
    biases = CustomMatrix(1, filter_quantity, true, 0.0f);
    float map_transform_matrix_values[16] {1, 0, -1, 0, 0, 1 , 1, 0, 0, -1, 1, 0, 0, 1, 0, -1};
    B_matrix.write(map_transform_matrix_values);
    float map_transform_matrix_values2[16] {1, 0, -1, 0, 0, 1 , 1, 0, 0, -1, 1, 0, 0, -1, 0, 1};
    B2_matrix.write(map_transform_matrix_values2);
    float filter_transform_matrix_values[12] {1, 0, 0, 0.5, 0.5, 0.5 , 0.5, -0.5, 0.5, 0, 0, 1};
    G_matrix.write(filter_transform_matrix_values);
    float flipped_filter_transform_matrix_values[8] {1, 0, 0.5, 0.5, 0.5, -0.5, 0, 1};
    G2_matrix.write(flipped_filter_transform_matrix_values);
    float inverse_transform_matrix_values[8] {1, 1, 1, 0, 0, 1, -1, -1};
    A_matrix.write(inverse_transform_matrix_values);
    float inverse_transform_matrix_values2[12] {1, 1, 1, 0, 0, 1, -1, 0, 0, 1, 1, 1};
    A2_matrix.write(inverse_transform_matrix_values2);
  }
  Convolutional(size_t p_height, size_t p_width, Activation act_p=relu):
  Layer{act_p}, map_dims {p_height, p_width}, filter_quantity {1}
  {
    biases = CustomMatrix(1, filter_quantity, true, 1.0f);
    float map_transform_matrix_values[16] {1, 0, -1, 0, 0, 1 , 1, 0, 0, -1, 1, 0, 0, 1, 0, -1};
    B_matrix.write(map_transform_matrix_values);
    float filter_transform_matrix_values[12] {1, 0, 0, 0.5, 0.5, 0.5 , 0.5, -0.5, 0.5, 0, 0, 1};
    G_matrix.write(filter_transform_matrix_values);
    float inverse_transform_matrix_values[8] {1, 1, 1, 0,  0, 1, -1, -1};
    A_matrix.write(inverse_transform_matrix_values);
    float inverse_transform_matrix_values2[12] {1, 1, 1, 0, 0, 1, -1, 0, 0, 1, 1, 1};
    A2_matrix.write(inverse_transform_matrix_values2);
    float map_transform_matrix_values2[16] {1, 0, -1, 0, 0, 1 , 1, 0, 0, -1, 1, 0, 0, -1, 0, 1};
    B2_matrix.write(map_transform_matrix_values2);
    float flipped_filter_transform_matrix_values[8] {1, 0, 0.5, 0.5, 0.5, -0.5, 0, 1};
    G2_matrix.write(flipped_filter_transform_matrix_values);
  }
  void forward(Tensor& input, cudaStream_t s)
  {

    map_transform<<<
      dim3(1, 1, input.depth * input.fourth),
      dim3(input.height / 2, input.width / 2)
    >>>(input, B_matrix, transformed_input);

    cudaDeviceSynchronize();

    filter_transform<<<
      1, 
      dim3(transformed_weights.height, transformed_weights.width, transformed_weights.depth * transformed_weights.fourth)
    >>>(weights, G_matrix, transformed_weights);
    cudaDeviceSynchronize();

    wts_input_mul_filter<<<
      dim3(1, 1, transformed_input.fourth * transformed_weights.fourth),
      dim3(transformed_input.height / 4, transformed_input.width / 4)
    >>>(transformed_input, transformed_weights, forward_inter);
    // wts = winograd transform space
    cudaDeviceSynchronize();

    inverse_transform_with_bias<<<
      dim3(1, 1, forward_inter.depth * forward_inter.fourth),
      dim3(forward_inter.height / 4, forward_inter.width / 4)
    >>>(forward_inter, A_matrix, biases, pre_activations);
    cudaDeviceSynchronize();

    conv_relu_kernel<<<
      dim3(1, 1, pre_activations.depth * pre_activations.fourth),
      dim3((pre_activations.height + 1) / 2, (pre_activations.width + 1) / 2)
    >>>(pre_activations, activations);

    cudaDeviceSynchronize();
  }

  void backward_conv(Tensor& nlw, Tensor& nle, cudaStream_t s)
  {
    flipped_filter_transform<<<
      1, 
      dim3(transformed_flipped_weights.height, transformed_flipped_weights.width, transformed_flipped_weights.depth * transformed_flipped_weights.fourth)
    >>>(nlw, G_matrix, transformed_flipped_weights);
    cudaDeviceSynchronize();

    map_transform<<<
      dim3(1, 1, nle.depth * nle.fourth),
      dim3(nle.height / 2, nle.width / 2)
    >>>(nle, B_matrix, transfromed_nle);
    cudaDeviceSynchronize();

    wts_nle_mul_nlw<<<
      dim3(1, 1, transfromed_nle.fourth * transformed_flipped_weights.depth),
      dim3(transfromed_nle.height / 4, transfromed_nle.width / 4)
    >>>(transfromed_nle, transformed_flipped_weights, backward_conv_inter);

    cudaDeviceSynchronize();
    
    inverse_transform<<<
      dim3(1, 1, backward_conv_inter.depth * backward_conv_inter.fourth),
      dim3(backward_conv_inter.height / 4, backward_conv_inter.width / 4)
    >>>(backward_conv_inter, A_matrix, errors);

    cudaDeviceSynchronize();

    conv_relu_derivative<<<
      dim3(1, 1, pre_activations.depth * pre_activations.fourth),
      dim3(pre_activations.height / 2, pre_activations.width / 2)
    >>>(pre_activations, pre_activations);

    cudaDeviceSynchronize();
    
    elementwisemul<<<
      get_grids(errors.height, errors.width, errors.depth * errors.fourth),
      get_threads(errors.height, errors.width)
    >>>(errors, pre_activations, errors);
    cudaDeviceSynchronize();
  }

  void backward_fc(Tensor& nlw, Tensor& nle, cudaStream_t s)
  {
    matmulmatTtoconv<<<
      get_grids(nle.height, get_output_size()), 
      get_threads(nle.height, get_output_size()),
      0,
      s
    >>>(nle, nlw, errors);
    cudaDeviceSynchronize();

    conv_relu_derivative<<<
      dim3(1, 1, pre_activations.depth * pre_activations.fourth),
      dim3(pre_activations.height / 2, pre_activations.width / 2),
      0,
      s
    >>>(pre_activations, pre_activations);
    cudaDeviceSynchronize();

    elementwisemul<<<
      get_grids(errors.height, errors.width, errors.depth * errors.fourth),
      get_threads(errors.height, errors.width),
      0,
      s
    >>>(errors, pre_activations, errors);
    cudaDeviceSynchronize();
  }

  void backward(Tensor& nlw, Tensor& nle, cudaStream_t s) // convback
  {
    bool is_next_layer_fcfc = (nle.fourth == 1);
    if (is_next_layer_fcfc)
    {
      backward_fc(nlw, nle, s);
    } else {
      backward_conv(nlw, nle, s);
    }
  }

  void update_weights(std::vector<std::reference_wrapper<Layer>>::iterator ll_iterator, float learning_rate, cudaStream_t s, bool use_alt=false)
  {

    // transform ll activations
    Tensor& ll_acts = use_alt ? ll_iterator->get().activations_alt : ll_iterator->get().activations;
    map_transform<<<
      dim3(1, 1, ll_acts.depth * ll_acts.fourth),
      dim3(ll_acts.height / 2, ll_acts.width / 2),
      0,
      s
    >>>(ll_acts, B2_matrix, ll_transformed_acts);
    cudaDeviceSynchronize();

    // transform errors
    map_transform_for_backprop<<<
      dim3(1, 1, errors.depth * errors.fourth),
      dim3(errors.height / 2, errors.width / 2),
      0,
      s
    >>>(errors, G2_matrix, transformed_errors);
    cudaDeviceSynchronize();

    // mul in wts with R * S reslut
    wts_ll_acts_mul_errs<<<
      dim3(1, 1, ll_transformed_acts.fourth * transformed_errors.depth * ll_transformed_acts.depth),
      dim3(ll_transformed_acts.height / 4, ll_transformed_acts.width / 4),
      0,
      s
    >>>(ll_transformed_acts, transformed_errors, weight_update_inter);
    cudaDeviceSynchronize();

    // inverse transform
    inverse_transform_for_weights<<<
      dim3(1, 1, weight_update_inter.depth * weight_update_inter.fourth),
      dim3(weight_update_inter.height / 4, weight_update_inter.width / 4),
      0,
      s
    >>>(weight_update_inter, A2_matrix, learning_rate / (float)ll_transformed_acts.fourth, weights);
    cudaDeviceSynchronize();

  }

  void set_input_props(const Layer& ll)
  {
    // std::cout << "conv set input props called \n";
    weights = Tensor(filter_dims[0], filter_dims[1], ll.get_depth(), filter_quantity);
    weight_update_inter = Tensor(4, 4, weights.depth, weights.fourth, true, 0);
    size_t tile_dim = weights.height + 2 - 1;
    transformed_weights = Tensor(tile_dim, tile_dim, ll.get_depth(), filter_quantity);
    transformed_input = Tensor(2 * ll.activations.height, 2 * ll.activations.width, ll.activations.depth, ll.activations.fourth);
    forward_inter = Tensor(2 * ll.activations.height, 2 * ll.activations.width, filter_quantity, ll.activations.fourth);
    ll_transformed_acts = Tensor (
      ll.activations.height * 2,
      ll.activations.width * 2,
      ll.activations.depth,
      ll.activations.fourth
    );
  }
  void initialize_with_batch_size(size_t batch_size, const Layer& ll)
  {
    if (&ll != this)
    {
      set_input_props(ll);
    }
    
    const size_t final_height = map_dims[0] ? map_dims[0] : ll.get_height();
    const size_t final_width = map_dims[1] ? map_dims[1] : ll.get_width();
    activations = Tensor(
      final_height, final_width, filter_quantity, batch_size);
    if (map_dims[0])
    {
      activations_alt = Tensor(
        final_height, final_width, filter_quantity, batch_size);
    } else {
      pre_activations = Tensor(
        final_height, final_width, filter_quantity, batch_size);
      errors = Tensor(
        final_height, final_width, filter_quantity, batch_size);
      transformed_errors = Tensor(
        errors.height * 2, errors.width * 2, errors.depth, errors.fourth
      );
    }
  }
  void initialize_with_next_layer(const Layer& nl)
  {
    transformed_flipped_weights = Tensor(nl.weights.height + 2 - 1, nl.weights.width + 2 - 1, nl.weights.depth, nl.weights.fourth);
    transfromed_nle = Tensor(nl.errors.height * 2, nl.errors.width * 2, nl.errors.depth, nl.errors.fourth);
    backward_conv_inter = Tensor(transfromed_nle.height, transfromed_nle.width, errors.depth, transfromed_nle.fourth);
  }
  size_t get_output_size() const
  {
    return activations.height * activations.width * activations.depth;
  }
  size_t get_output_bias_size() const 
  {
    return 0;
  }
  size_t get_depth() const
  {
    // return activations.depth;
    return filter_quantity;
  }
  size_t get_height() const
  {
    return activations.height;
  }
  size_t get_width() const
  {
    return activations.width;
  }
};

typedef void (*out_err_fptr)(Tensor, Tensor, int*);

out_err_fptr get_out_err_func(
  void (*out_loss)(float*, float*, size_t),
  void (*out_act)(Tensor, Tensor))
{
  if (out_loss == cross_entropy)
  {
    if (out_act == softmax_kernel)
    {
      return softmax_crossen_error;
    }else{
      return nullptr;
    }
  }else if (out_loss == mean_square_error){
    if (false)//out_act == sigmoid_kernel
    {
      return sigmoid_square_error;
    }else{
      return nullptr;
    }
  }else{
    return nullptr;
  }
}

class Model
{
public:
  std::vector<std::reference_wrapper<Layer>> layers {};
  int* d_correct_labels {};
  int* d_correct_labels_alt {};
  void (*loss_func)(float*, float*, size_t);
  void (*out_err_func)(Tensor, Tensor, int*);
  bool final {false};
  float learning_rate;
  int* d_correct_predictions {};
  cudaStream_t data_transfer_s;
  cudaStream_t kernel_exec_s;

  Model(void (*p_loss_func)(float*, float*, size_t), float p_learning_rate):
  loss_func{p_loss_func}, learning_rate{p_learning_rate}
  {
    cudaMalloc((void **) &d_correct_predictions, sizeof(int));
    cudaStreamCreate(&data_transfer_s);
    cudaStreamCreate(&kernel_exec_s);
  }
  void reset_correct_predictions()
  {
    int zero {0};
    cudaMemcpy(d_correct_predictions, &zero, sizeof(int), cudaMemcpyHostToDevice);
  }
  int read_correct_predictions()
  {
    int ans {};
    cudaMemcpy(&ans, d_correct_predictions, sizeof(int), cudaMemcpyDeviceToHost);
    return ans;
  }
  bool finalize(size_t batch_size)
  {
    if (get_out_err_func(loss_func, layers.back().get().act.f))
    {
      out_err_func = get_out_err_func(loss_func, layers.back().get().act.f);
      for (std::vector<std::reference_wrapper<Layer>>::iterator l = layers.begin(); l != layers.end(); ++l)
      {
        l->get().initialize_with_batch_size(batch_size, l == layers.begin() ? l->get() : (l - 1)->get());
      }
      for (std::vector<std::reference_wrapper<Layer>>::iterator l = layers.begin(); l != layers.end(); ++l)
      {
        l->get().initialize_with_next_layer(l == layers.end() - 1 ? l->get() : (l + 1)->get());
      }
      cudaMalloc((void**) &d_correct_labels, sizeof(int) * batch_size);
      cudaMalloc((void**) &d_correct_labels_alt, sizeof(int) * batch_size);
      final = true;
      return true;
    }
    return false;
  }
  void add(Layer& l)
  {
    if(!layers.empty())
    {
      l.set_input_props(layers.back().get());
      layers.push_back(l);
    }else{
      layers.push_back(l);
    }
  }
  void move_batch(float* input_data, int* targets, size_t batch_size, bool use_alt)
  {
    cudaMemcpy2DAsync(
      (use_alt ? layers.front().get().activations_alt.d_copy : layers.front().get().activations.d_copy),
      (use_alt ? layers.front().get().activations_alt.pitch : layers.front().get().activations.pitch),
      input_data, 
      sizeof(float) * (layers.front().get().activations.width),
      sizeof(float) * (layers.front().get().activations.width),
      layers.front().get().activations.CustomMatrix::height,
      cudaMemcpyHostToDevice,
      data_transfer_s);
    cudaMemcpyAsync(
      (use_alt ? d_correct_labels_alt : d_correct_labels), 
      targets, 
      sizeof(int) * batch_size, 
      cudaMemcpyHostToDevice,
      data_transfer_s);
  }
  void forward_pass(size_t batch_size, bool use_alt)
  {
    Tensor temp_results = (use_alt ? layers.front().get().activations_alt : layers.front().get().activations);
    for (std::vector<std::reference_wrapper<Layer>>::iterator l = layers.begin() + 1; l != layers.end(); ++l)
    {
      l->get().forward(temp_results, kernel_exec_s);
      temp_results = l->get().activations;
    }
    update_correct_labels<<<1, batch_size, 0, kernel_exec_s>>>(
      layers.back().get().activations, 
      (use_alt ? d_correct_labels_alt : d_correct_labels), 
      d_correct_predictions);
  }
  void backprop(size_t batch_size, bool use_alt)
  {
    out_err_func<<<
	    get_grids(batch_size, layers.back().get().get_output_size()), 
	    get_threads(batch_size, layers.back().get().get_output_size()), 
	    0, 
	    kernel_exec_s
    >>>(
    	layers.back().get().activations, 
	    layers.back().get().errors, 
	    (use_alt ? d_correct_labels_alt : d_correct_labels)
	  );
    for (std::vector<std::reference_wrapper<Layer>>::iterator l = layers.end() - 2; l != layers.begin(); --l)
    {
      l->get().backward((l + 1)->get().weights, (l + 1)->get().errors, kernel_exec_s);
    }
  }
  void weight_update(bool use_alt)
  {
    // Tensor& input_activations = (use_alt ? layers[0].get().activations_alt : layers[0].get().activations);
    // weight_update_kernel_for_conv<<<
    //   get_grids(layers[1].get().weights.height, layers[1].get().weights.width),
    //   get_threads(layers[1].get().weights.height, layers[1].get().weights.width),
    //   0, 
    //   kernel_exec_s
    // >>>(layers[1].get().errors, input_activations, layers[1].get().weights, learning_rate);
    std::vector<std::reference_wrapper<Layer>>::iterator first_layer = layers.begin();
    layers[1].get().update_weights(first_layer, learning_rate, kernel_exec_s, use_alt);

    for (std::vector<std::reference_wrapper<Layer>>::iterator l = layers.begin() + 2; l != layers.end(); ++l)
    {
      // weight_update_kernel<<<
      //   get_grids(l->get().weights.height, l->get().weights.width),
      //   get_threads(l->get().weights.height, l->get().weights.width),
      //   0, 
      //   kernel_exec_s
      // >>>(l->get().errors, (l - 1)->get().activations, l->get().weights, learning_rate); 
      l->get().update_weights(l - 1, learning_rate, kernel_exec_s);
    }
  }
  void single_train_timed(float* image, int* label, size_t batch_size)
  {
    auto t0 = std::chrono::high_resolution_clock::now();
    move_batch(image, label, batch_size, false);
    cudaDeviceSynchronize();
    auto t1 = std::chrono::high_resolution_clock::now();
    forward_pass(batch_size, false);
    auto t2 = std::chrono::high_resolution_clock::now();
    backprop(batch_size, false);
    auto t3 = std::chrono::high_resolution_clock::now();
    weight_update(false);
    cudaDeviceSynchronize();
    auto t4 = std::chrono::high_resolution_clock::now();
    std::chrono::nanoseconds move_time = t1 - t0;
    std::chrono::nanoseconds forward_time = t2 - t1;
    std::chrono::nanoseconds back_time = t3 - t2;
    std::chrono::nanoseconds update_time = t4 - t1;
    std::cout << move_time.count() << "ns \n";
    std::cout << forward_time.count() << "ns \n";
    std::cout << back_time.count() << "ns \n";
    std::cout << update_time.count() << "ns \n";
  }
  void single_train(float* image, int* label, size_t batch_size)
  {
    move_batch(image, label, batch_size, false);
    cudaDeviceSynchronize();
    forward_pass(batch_size, false);
    cudaDeviceSynchronize();
    backprop(batch_size, false);
    cudaDeviceSynchronize();
    weight_update(false);
    cudaDeviceSynchronize();
  }
  void single_test(float* image, int* label, size_t batch_size)
  {
    move_batch(image, label, batch_size, false);
    cudaDeviceSynchronize();
    forward_pass(batch_size, false);
    cudaDeviceSynchronize();
  }
  template <typename T1, typename T2, size_t S, size_t item_length1, size_t item_length2>
  void train_sequential(
    PinnedData<T1, S, item_length1> images,
    PinnedData<T2, S, item_length2> labels,
    int epochs,
    size_t batch_size)
  {
    if (finalize(batch_size))
    {
      for (int epoch = 1; epoch <= epochs; ++epoch)
      {
        reset_correct_predictions();
        int num_of_data = images.size;
        auto tik = std::chrono::high_resolution_clock::now();
        for (int loopIdx = 0; loopIdx < num_of_data; loopIdx += batch_size)
        {
          single_train(images[loopIdx], labels[loopIdx], batch_size);
        }
        auto tok = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double, std::milli> ms_double = tok - tik;
        std::cout << "Epoch " << epoch << " : acc = ";
        std::cout << read_correct_predictions()/(float)num_of_data; 
        std::cout << " in " << ms_double.count() << "ms.\n"; 
      }
    }else{
      std::cout << "Could not finalize model. \n";
    }
  }
  template <typename T1, typename T2, size_t S, size_t item_length1, size_t item_length2>
  void train(
    PinnedData<T1, S, item_length1> images,
    PinnedData<T2, S, item_length2> labels,
    int epochs,
    size_t batch_size)
  {
    if (finalize(batch_size))
    {
      for (int epoch = 1; epoch <= epochs; ++epoch)
      {
        bool use_alt {false};
        reset_correct_predictions();
        cudaDeviceSynchronize();
        int num_of_data = images.size;
        auto tik = std::chrono::high_resolution_clock::now();
        move_batch(images[0], labels[0], batch_size, use_alt);
        cudaDeviceSynchronize();
        for (int loopIdx = batch_size; loopIdx < num_of_data; loopIdx += batch_size)
        {
          move_batch(
            images[loopIdx], 
            labels[loopIdx], 
            batch_size, 
            !use_alt);
          cudaDeviceSynchronize();
          forward_pass(batch_size, use_alt);
          cudaDeviceSynchronize();
          backprop(batch_size, use_alt);
          cudaDeviceSynchronize();
          weight_update(use_alt);
          cudaDeviceSynchronize();
          use_alt = !use_alt;
        }
        auto tok = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double, std::milli> ms_double = tok - tik;
        std::cout << "Epoch " << epoch << " : acc = ";
        std::cout << read_correct_predictions()/(float)num_of_data; 
        std::cout << " in " << ms_double.count() << "ms.\n"; 
      }
    }else{
      std::cout << "Could not finalize model. \n";
    }
  }
  template <typename T1, typename T2, size_t S, size_t item_length1, size_t item_length2>
  void test(
    PinnedData<T1, S, item_length1> images,
    PinnedData<T2, S, item_length2> labels,
    size_t batch_size)
  {
    int num_of_data = images.size;
    reset_correct_predictions();
    for (int loopIdx = 0; loopIdx < num_of_data; loopIdx += batch_size)
    {
      single_test(images[loopIdx], labels[loopIdx], batch_size);
    }
    std::cout << "test acc = " << read_correct_predictions()/(float)num_of_data << '\n';
  }
};

__global__ void test(int a, int* b)
{
  *b = a + 1;
}
void testCuda()
{
  int* dp;
  int hp;
  cudaMalloc((void**) &dp, sizeof(int));
  test<<<1, 1>>>(1, dp);
  cudaMemcpy(&hp, dp, sizeof(int), cudaMemcpyDeviceToHost);
  if(hp != 2)
  {
    std::cout << "Cuda Not Working. Factory Reset Runtime. \n";
  }
}
void print_column_np(Tensor& tens, int column=0)
{
  float* result = tens.read();
  std::cout << '[';
  for (int j = 0; j < tens.height; ++j)
  {
    std::cout << result[j * tens.width + column] << ", ";
  }
  std::cout << "] \n";
}

int main()
{
  testCuda();

  std::srand(0);//static_cast<unsigned int>(std::time(nullptr))
  std::rand(); 

  // PinnedData<float, 10000, 784> test_images("sample_data/mnist_test.csv", false);
  // PinnedData<int, 10000, 1> test_labels("sample_data/mnist_test.csv");
  // PinnedData<float, 20000, 784> train_images("sample_data/mnist_train_small.csv", false);
  // PinnedData<int, 20000, 1> train_labels("sample_data/mnist_train_small.csv");
  PinnedData<float, 10000, 784> test_images("../input/mnistdata/mnist_test.csv", false);
  PinnedData<int, 10000, 1> test_labels("../input/mnistdata/mnist_test.csv");
  PinnedData<float, 20000, 784> train_images("../input/mnistdata/mnist_train_small.csv", false);
  PinnedData<int, 20000, 1> train_labels("../input/mnistdata/mnist_train_small.csv");

  // std::cout << "config: layer1:C28*28, layer2:C5filters3*3, layer3:R128, layer4:R10Softmax, lr=0.05, commit_hash:ea1472, env:kaggle-MFP, GPU:Tesla P100-PCIE-16GB" << '\n';

  // auto layer1 = Regular(784, relu, true);
  auto layer1 = Convolutional(28, 28);
  // auto layer1 = Convolutional(5, 5);
  
  // auto layer2 = Regular(32);
  // auto layer2 = FCfromConv(128);
  auto layer2 = Convolutional(5, {3, 3});

  // auto layer3 = Regular(32);
  auto layer3 = FCfromConv(16);
  // auto layer3 = Convolutional(2, {4, 4});

  // auto layer4 = FCfromConv(10, softmax);
  auto layer4 = Regular(10, softmax);

  Model mnist_model(cross_entropy, 0.05f);
  // Model mnist_model(cross_entropy, 2.0f);
  mnist_model.add(layer1);
  mnist_model.add(layer2);
  mnist_model.add(layer3);
  mnist_model.add(layer4);

  size_t mini_batch_size {4};

  mnist_model.finalize(mini_batch_size);

  auto tik = std::chrono::high_resolution_clock::now();
  mnist_model.train_sequential(train_images, train_labels, 1, mini_batch_size);

  auto tok = std::chrono::high_resolution_clock::now();
  std::chrono::duration<double, std::milli> ms_double = tok - tik;
  std::cout << ms_double.count() << "ms \n";


  mnist_model.layers[1].get().weights.make_file("probs_broken_l2_weights.t");
  // std::cout << "layer 2 weights before: \n";
  // std::cout << mnist_model.layers[1].get().weights << '\n';
  // std::cout << "layer 3 weights before: \n";
  // std::cout << mnist_model.layers[2].get().weights << '\n';
  // std::cout << "layer 4 weights before: \n";
  // std::cout << mnist_model.layers[3].get().weights << '\n';
  // mnist_model.move_batch(train_images[0], train_labels[0], mini_batch_size, false);
  // cudaDeviceSynchronize();
  // std::cout << cudaGetErrorName(cudaPeekAtLastError()) << '\n';
  // mnist_model.forward_pass(mini_batch_size, false);
  // cudaDeviceSynchronize();
  // std::cout << cudaGetErrorName(cudaPeekAtLastError()) << '\n';
  // mnist_model.backprop(mini_batch_size, false);
  // cudaDeviceSynchronize();
  // std::cout << cudaGetErrorName(cudaPeekAtLastError()) << '\n';
  // mnist_model.layers[1].get().errors.make_file("l2_errs.t");
  // mnist_model.layers[0].get().activations.make_file("l1_acts.t");
  // mnist_model.weight_update(false);
  // cudaDeviceSynchronize();
  // std::cout << cudaGetErrorName(cudaPeekAtLastError()) << '\n';
  // mnist_model.layers[1].get().weights.make_file("updated_l2_weights.t");
  // std::cout << "layer 1 act: \n";
  // std::cout << mnist_model.layers[0].get().activations << '\n';
  // std::cout << "layer 2 act: \n";
  // std::cout << mnist_model.layers[1].get().activations << '\n';
  // std::cout << "layer 3 act: \n";
  // std::cout << mnist_model.layers[2].get().activations << '\n';
  // std::cout << "layer 4 act: \n";
  // std::cout << mnist_model.layers[3].get().pre_activations << '\n';
  // std::cout << "layer 2 errs: \n";
  // std::cout << mnist_model.layers[1].get().errors << '\n';
  // std::cout << "layer 3 errs: \n";
  // std::cout << mnist_model.layers[2].get().errors << '\n';
  // std::cout << "layer 4 errs: \n";
  // std::cout << mnist_model.layers[3].get().errors << '\n';
  // std::cout << "layer 2 weights: \n";
  // std::cout << mnist_model.layers[1].get().weights << '\n';
  // std::cout << "layer 3 weights: \n";
  // std::cout << mnist_model.layers[2].get().weights << '\n';
  // std::cout << "layer 4 weights: \n";
  // std::cout << mnist_model.layers[3].get().weights << '\n';
  
  // auto tik = std::chrono::high_resolution_clock::now();
  // mnist_model.train(train_images, train_labels, 7, mini_batch_size);

  // auto tok = std::chrono::high_resolution_clock::now();
  // std::chrono::duration<double, std::milli> ms_double = tok - tik;
  // std::cout << ms_double.count() << "ms \n";

  // mnist_model.test(test_images, test_labels, mini_batch_size);

  return 0;
}
