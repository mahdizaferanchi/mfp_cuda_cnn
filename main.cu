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


template <class T, size_t S, size_t item_length>
class pinned_data
{
public:
	T* beginning {};
	T* end;
	size_t size = S;
	pinned_data(const std::string& file_name)
	{
		cudaMallocHost((void**) &beginning, sizeof(T) * size * item_length, 4);
		end = beginning;
		std::ifstream file(file_name);
		std::string data_point_string;
		if (item_length == 1)
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
		for (int i = 0; i < 784; ++i)
		{
			end[i] = static_cast<float> (std::stoi(nums[i + 1])) / 255.0;
		}
		end[784] = 1.0f;
		end += 785;
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
		return &beginning[idx * item_length];
	}
};

class mnist_data_point
{
public:
	float image[784];
	int label;
	mnist_data_point(std::string& str)
	{
		std::stringstream line(str);
		std::string num;
		std::vector<std::string> nums;
		while(std::getline(line, num, ','))
		{
			nums.push_back(num);
		}
		label = std::stoi(nums[0]);
		for (int i = 0; i < 784; ++i)
		{
			image[i] = static_cast<float> (std::stoi(nums[i + 1])) / 255.0;
		}
	}
};
class mnist_image
{
public:
	float image[785];
	mnist_image(std::string& str)
	{
		cudaMallocHost((void**) &image, 785 * sizeof(float), 4);
		std::stringstream line(str);
		std::string num;
		std::vector<std::string> nums;
		while(std::getline(line, num, ','))
		{
			nums.push_back(num);
		}
		for (int i = 0; i < 784; ++i)
		{
			image[i] = static_cast<float> (std::stoi(nums[i + 1])) / 255.0;
		}
		image[784] = 1.0f;
	}
};

class mnist_label
{
public:
	int label[1];
	mnist_label(std::string& str)
	{
		// cudaError_t status = cudaMallocHost((void**) &label, sizeof(int), 4);
		std::stringstream line(str);
		std::string num;
		std::getline(line, num, ',');
		label[0] = std::stoi(num);
	}
};

std::vector<mnist_data_point> mnist_parse(const std::string& file_name)
{
	std::ifstream file(file_name);
	std::vector<mnist_data_point> data_vector;
	std::string data_point_string;
	while(std::getline(file, data_point_string))
	{
		mnist_data_point p(data_point_string);
		data_vector.push_back(p);
	}
	return data_vector;
}

std::vector<mnist_image> mnist_parse_image(const std::string& file_name)
{
	std::ifstream file(file_name);
	std::vector<mnist_image> data_vector;
	std::string data_point_string;
	while(std::getline(file, data_point_string))
	{
		mnist_image p(data_point_string);
		data_vector.push_back(p);
	}
	return data_vector;
}

std::vector<mnist_label> mnist_parse_label(const std::string& file_name)
{
	std::ifstream file(file_name);
	std::vector<mnist_label> data_vector;
	std::string data_point_string;
	while(std::getline(file, data_point_string))
	{
		mnist_label p(data_point_string);
		data_vector.push_back(p);
	}
	return data_vector;
}

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

class c_vector
{
public:
	size_t length;
	float* h_copy;
	float* d_copy{};
	c_vector(size_t p_size, float initial_val=1):
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
	friend std::ostream& operator<<(std::ostream& os, c_vector& vec)
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

class c_matrix
{
public:
	size_t pitch;
	size_t height;
	size_t width;
	float* h_copy;
	float* d_copy{};
	c_matrix(size_t p_height, size_t p_width, bool one_initialization=false, float initial_max=0.5):
	height{p_height}, width{p_width}
	{
		size_t length = height * width;
		size_t float_size = sizeof(float);
		h_copy = new float[length];
		cudaMallocPitch((void**) &d_copy, &pitch, float_size * width, height);
		if (!one_initialization)
		{
			fill_with_rand(h_copy, length, initial_max);
			for (int i = 0; i < width; ++i)
			{
				h_copy[(height - 1) * width + i] = 0;
			}
		}else{
			std::fill_n(h_copy, length, 1.0f);
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
	friend std::ostream& operator<<(std::ostream& os, c_matrix& mat)
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

class tensor_4d : public c_matrix
{
public:
	size_t height;
	size_t width;
	size_t depth;
	size_t fourth;
	tensor_4d(size_t p_height, size_t p_width, size_t p_depth=1, size_t p_fourth=1, 
		bool one_initialization=false, float initial_max=0.5):
	c_matrix{p_height * p_depth * p_fourth, p_width, one_initialization, initial_max},
	height {p_height}, width {p_width}, depth {p_depth}, fourth {p_fourth}
	{}
	friend std::ostream& operator<<(std::ostream& os, tensor_4d& mat)
	{
		float* result = mat.read();
		for (int f = 0; f < mat.fourth; ++f)
		{
			os << "block " << f << '\n';
			for (int d = 0; d < mat.depth; ++d)
			{
				os << "depth " << d << '\n';
				for (int i = 0; i < mat.height; ++i)
				{
					for (int j = 0; j < mat.width; ++j)
					{
						os << result[f * mat.depth * mat.height + d * mat.height + i * mat.width + j] << " ";
					}
					os << '\n';
				}	
			}
		}
		return os;
	}
	__device__ inline float* at(int row, int col, int page=1, int block=1)
	{
		return (float*)((char*)d_copy + (block * height * depth + page * height + row) * pitch) + col;
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

__global__ void matmulmat(tensor_4d left, tensor_4d right, tensor_4d out)
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

__global__ void matmulmatT(tensor_4d left, tensor_4d right, tensor_4d out)
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

__global__ void relu_kernel(tensor_4d in, tensor_4d out)
{
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	int j = blockIdx.y * blockDim.y + threadIdx.y;

	// if (in.at(i, j) < 0)
	// {
	// 	out.at(i, j) = 0;
	// }else{
	// 	out.at(i, j) = in.at(i, j);
	// }
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

__global__ void elementwisemul(tensor_4d left, tensor_4d right, tensor_4d out)
{
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	int j = blockIdx.y * blockDim.y + threadIdx.y;

	if (i < left.height && j < left.width)
		*out.at(i, j) = *left.at(i, j) * (*right.at(i, j));
}

__global__ void relu_derivative(tensor_4d in, tensor_4d out)
{
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	int j = blockIdx.y * blockDim.y + threadIdx.y;
	// if (i < size)
	// {
	// 	if (input[i] < 0)
	// 	{
	// 		output[i] = 0;
	// 	}else{
	// 		output[i] = 1;
	// 	}
	// }
	if (i < in.height && j < in.width)
		*out.at(i, j) = (*in.at(i, j) < 0.0f) ? 0.0f : 1.0f;
}

__global__ void softmax_kernel(tensor_4d in, tensor_4d out)
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

__global__ void softmax_crossen_error(tensor_4d in, tensor_4d out, int* targets)
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

__global__ void sigmoid_square_error(tensor_4d in, tensor_4d out, int* targets)
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

__global__ void weight_update_kernel(tensor_4d errors, tensor_4d last_activations, tensor_4d weights, float learning_rate)
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

__global__ void update_correct_labels(tensor_4d acts, int* labels, int* correct_predictions)
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

class activation
{
public:
	void (*f)(tensor_4d, tensor_4d);
	void (*d)(tensor_4d, tensor_4d);
	activation(void (*p_f)(tensor_4d, tensor_4d), void (*p_d)(tensor_4d, tensor_4d)):
	f{p_f}, d{p_d}
	{} 
};

activation relu(relu_kernel, relu_derivative);
activation softmax(softmax_kernel, relu_derivative);
// activation sigmoid(sigmoid_kernel, sigmoid_derivative);

dim3 get_grids(size_t x_dim, size_t y_dim)
{
	return dim3((x_dim > 40) ? x_dim / 20 + 1 : 2, (y_dim > 40) ? y_dim / 20 + 1 : 2);
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
	tensor_4d activations {1, 1};
	tensor_4d activations_alt {1, 1};
	tensor_4d pre_activations {1, 1};
	tensor_4d errors {1, 1};
	tensor_4d weights {1, 1};
	activation act;
	Layer(activation act_p):
	act {act_p}
	{}
	virtual void forward(tensor_4d& input, cudaStream_t s) = 0;
	virtual void backward(tensor_4d& nlw, tensor_4d& nle, cudaStream_t s) = 0;
	virtual void set_input_props(const Layer& lla) = 0;
	virtual void initialize_with_batch_size(size_t batch_size, const Layer& ll) = 0;
	virtual size_t get_output_size() const = 0;
	virtual size_t get_depth() const = 0;
	virtual size_t get_height() const = 0;
	virtual size_t get_width() const = 0;
};


class Regular : public Layer
{
public:
	size_t units;
	size_t input_length;
	// tensor_4d activations {1, 1};
	// tensor_4d activations_alt {1, 1};
	// tensor_4d pre_activations {1, 1};
	// tensor_4d errors {1, 1};
	// tensor_4d weights {1, 1};
	// activation act;
	bool double_activations;
	Regular(size_t p_units=16, activation act_p=relu, bool p_double_activations=false, size_t p_input_length=1):
	Layer{act_p}, units{p_units}, input_length{p_input_length}, double_activations{p_double_activations}
	{}
	void forward(tensor_4d& input, cudaStream_t s)
	{
		matmulmat<<<
			get_grids(input.height, units),
			get_threads(input.height, units),
			0, 
			s>>>
			(input, weights, pre_activations);
		act.f<<<
			get_grids(input.height, units),
			get_threads(input.height, units), 
			0, 
			s>>>
			(pre_activations, activations);
	}
	void backward(tensor_4d& nlw, tensor_4d& nle, cudaStream_t s)
	{
		matmulmatT<<<
			get_grids(nle.height, units), 
			get_threads(nle.height, units), 
			0, 
			s>>>
			(nle, nlw, errors);
		act.d<<<
			get_grids(pre_activations.height, units), 
			get_threads(pre_activations.height, units), 
			0, 
			s>>>
			(pre_activations, pre_activations);
		elementwisemul<<<
			get_grids(errors.height, units), 
			get_threads(errors.height, units), 
			0, 
			s>>>
			(errors, pre_activations, errors);
	}
	void set_input_props(const Layer& ll)
	{
		input_length = ll.get_output_size();
		weights = tensor_4d(input_length, units);
	}
	void initialize_with_batch_size(size_t batch_size, const Layer& ll)
	{
		activations = tensor_4d(batch_size, units + 1, true);
		pre_activations = tensor_4d(batch_size, units, true);
		errors = tensor_4d(batch_size, units, true);
		if (double_activations)
			activations_alt = tensor_4d(batch_size, units + 1, true);
	}
	size_t get_output_size() const
	{
		return units + 1;
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

class Convolutional : public Layer
{
public:
	size_t filter_quantity;
	std::array<size_t, 2> filter_dims;
	bool same_padding {true};
	// tensor_4d activations {1, 1, 1, 1};
	// tensor_4d pre_activations {1, 1, 1, 1};
	// tensor_4d errors {1, 1, 1, 1};
	// tensor_4d weights {1, 1, 1, 1};
	// activation act;
	Convolutional(size_t p_filter_quantity, std::array<size_t, 2> p_filter_dims,
		activation act_p=relu, bool p_same_padding=true):
	Layer{act_p}, filter_quantity {p_filter_quantity}, same_padding {p_same_padding},
	filter_dims {p_filter_dims}
	{}
	void set_input_props(const Layer& ll)
	{
		// Convolutional& lc = static_cast<Convolutional&>(ll);
		weights = tensor_4d(filter_dims[0], filter_dims[1], ll.get_depth(), filter_quantity);
	}
	void initialize_with_batch_size(size_t batch_size, const Layer& ll)
	{
		// Convolutional& lc = static_cast<Convolutional&>(ll);
		activations = tensor_4d(
			ll.get_height(), ll.get_width(), filter_quantity, batch_size);
		pre_activations = tensor_4d(
			ll.get_height(), ll.get_width(), filter_quantity, batch_size);
		errors = tensor_4d(
			ll.get_height(), ll.get_width(), filter_quantity, batch_size);
	}
	size_t get_output_size() const
	{
		return activations.height * activations.width * activations.depth * activations.fourth;
	}
	size_t get_depth() const
	{
		return weights.depth;
	}
	size_t get_height() const
	{
		return weights.height;
	}
	size_t get_width() const
	{
		return weights.width;
	}
};

typedef void (*out_err_fptr)(tensor_4d, tensor_4d, int*);

out_err_fptr get_out_err_func(
	void (*out_loss)(float*, float*, size_t),
	void (*out_act)(tensor_4d, tensor_4d))
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

class model
{
public:
	std::vector<std::reference_wrapper<Layer>> layers {};
	int* d_correct_labels {};
	int* d_correct_labels_alt {};
	void (*loss_func)(float*, float*, size_t);
	void (*out_err_func)(tensor_4d, tensor_4d, int*);
	bool final {false};
	float learning_rate;
	int* d_correct_predictions {};
	cudaStream_t data_transfer_s;
	cudaStream_t kernel_exec_s;

	model(void (*p_loss_func)(float*, float*, size_t), float p_learning_rate):
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
			// for (int loopIdx = 0; loopIdx < layers.size(); ++loopIdx)
			// {
			// 	layers[loopIdx].initialize_with_batch_size(batch_size);	
			// }
			for (std::vector<std::reference_wrapper<Layer>>::iterator l = layers.begin(); l != layers.end(); ++l)
			{
				l->get().initialize_with_batch_size(batch_size, l == layers.begin() ? l->get() : (l - 1)->get());
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

		// auto t0 = std::chrono::high_resolution_clock::now();
		cudaMemcpy2DAsync(
			(use_alt ? layers.front().get().activations_alt.d_copy : layers.front().get().activations.d_copy),
			(use_alt ? layers.front().get().activations_alt.pitch : layers.front().get().activations.pitch),
			input_data, 
			sizeof(float) * (layers.front().get().get_output_size()),
			sizeof(float) * (layers.front().get().get_output_size()),
			batch_size,
			cudaMemcpyHostToDevice,
			data_transfer_s);
		// auto t1 = std::chrono::high_resolution_clock::now();
		// std::chrono::nanoseconds dt = t1 - t0;
		// std::cout << dt.count() << '\n';
		cudaMemcpyAsync(
			(use_alt ? d_correct_labels_alt : d_correct_labels), 
			targets, 
			sizeof(int) * batch_size, 
			cudaMemcpyHostToDevice,
			data_transfer_s);
	}
	void forward_pass(size_t batch_size, bool use_alt)
	{
		tensor_4d temp_results = (use_alt ? layers.front().get().activations_alt : layers.front().get().activations);
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
		get_grids(batch_size, layers.back().get().get_output_size() - 1), 
		get_threads(batch_size, layers.back().get().get_output_size() - 1), 
		0, 
		kernel_exec_s>>>
		(layers.back().get().activations, 
		layers.back().get().errors, 
		(use_alt ? d_correct_labels_alt : d_correct_labels));
		for (std::vector<std::reference_wrapper<Layer>>::iterator l = layers.end() - 2; l != layers.begin(); --l)
		{
			l->get().backward((l + 1)->get().weights, (l + 1)->get().errors, kernel_exec_s);
		}
	}
	void weight_update(bool use_alt)
	{
		// tensor_4d& input_activations = (use_alt ? layers[0].activations_alt : layers[0].activations);
		// // int dim = (layers[1].weights.height > 50) ? 20 : 2;
		// weight_update_kernel<<<
		// 	get_grids(layers[1].weights.height, layers[1].weights.width),
		// 	get_threads(layers[1].weights.height, layers[1].weights.width),
		// 	0, 
		// 	kernel_exec_s>>>
		// 	(layers[1].errors, input_activations, layers[1].weights, learning_rate);
		// for (std::vector<Layer>::iterator l = layers.begin() + 2; l != layers.end(); ++l)
		// {
		// 	// int var = (l->weights.height > 50) ? 20 : 2;
		// 	weight_update_kernel<<<
		// 		get_grids(l->weights.height, l->weights.width),
		// 		get_threads(l->weights.height, l->weights.width),
		// 		0, 
		// 		kernel_exec_s>>>
		// 		(l->errors, (l - 1)->activations, l->weights, learning_rate); 
		// }
	}
	void single_train_timed(float* image, int* label, size_t batch_size)
	{
		auto t0 = std::chrono::high_resolution_clock::now();
		move_batch(image, label, batch_size, false);
		cudaDeviceSynchronize();
		auto t1 = std::chrono::high_resolution_clock::now();
		forward_pass(batch_size, false);
		// auto t2 = std::chrono::high_resolution_clock::now();
		backprop(batch_size, false);
		// auto t3 = std::chrono::high_resolution_clock::now();
		weight_update(false);
		cudaDeviceSynchronize();
		auto t4 = std::chrono::high_resolution_clock::now();
		std::chrono::nanoseconds move_time = t1 - t0;
		// std::chrono::nanoseconds forward_time = t2 - t1;
		// std::chrono::nanoseconds back_time = t3 - t2;
		std::chrono::nanoseconds update_time = t4 - t1;
		std::cout << move_time.count() << "ns \n";
		// std::cout << forward_time.count() << "ns \n";
		// std::cout << back_time.count() << "ns \n";
		std::cout << update_time.count() << "ns \n";
	}
	void single_train(float* image, int* label, size_t batch_size)
	{
		move_batch(image, label, batch_size, false);
		cudaDeviceSynchronize();
		forward_pass(batch_size, false);
		backprop(batch_size, false);
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
	void train(
		pinned_data<T1, S, item_length1> images,
		pinned_data<T2, S, item_length2> labels,
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
	void train_pipelined(
		pinned_data<T1, S, item_length1> images,
		pinned_data<T2, S, item_length2> labels,
		int epochs,
		size_t batch_size)
	{
		if (finalize(batch_size))
		{
			for (int epoch = 1; epoch <= epochs; ++epoch)
			{
				bool use_alt {false};
				reset_correct_predictions();
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
					// cudaDeviceSynchronize();
					forward_pass(batch_size, use_alt);
					backprop(batch_size, use_alt);
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
		pinned_data<T1, S, item_length1> images,
		pinned_data<T2, S, item_length2> labels,
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

int main()
{
	std::srand(0);//static_cast<unsigned int>(std::time(nullptr))
	std::rand(); 

	tensor_4d sample_tensor {2, 3, 3, 2};
	std::cout << sample_tensor << '\n';

	// pinned_data<float, 10000, 785> test_images("sample_data/mnist_test.csv");
	// pinned_data<int, 10000, 1> test_labels("sample_data/mnist_test.csv");
	// pinned_data<float, 20000, 785> train_images("sample_data/mnist_train_small.csv");
	// pinned_data<int, 20000, 1> train_labels("sample_data/mnist_train_small.csv");

	Regular layer1 = Regular(784, relu, true);
	Regular layer2 = Regular(32);
	Regular layer3 = Regular(32);
	Regular layer4 = Regular(10, softmax);

	model mnist_model(cross_entropy, 0.05f);
	mnist_model.add(layer1);
	mnist_model.add(layer2);
	mnist_model.add(layer3);
	mnist_model.add(layer4);

	mnist_model.finalize(32);

	// auto tik = std::chrono::high_resolution_clock::now();
	// mnist_model.train_pipelined(train_images, train_labels, 7, 32);

	// // mnist_model.learning_rate = 0.001f;
	// // mnist_model.train_pipelined(train_images, train_labels, 5, 32);

	// // mnist_model.learning_rate = 0.0001f;
	// // mnist_model.train_pipelined(train_images, train_labels, 5, 32);

	// auto tok = std::chrono::high_resolution_clock::now();
	// std::chrono::duration<double, std::milli> ms_double = tok - tik;
	// std::cout << ms_double.count() << "ms \n";

	// mnist_model.test(test_images, test_labels, 32);

	return 0;
}
