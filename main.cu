#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include <sstream>
#include <random>
#include <ctime>
#include <algorithm>
#include <chrono>



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
			image[i] = static_cast<float> (std::stoi(nums[i + 1])) / 256.0;
		}
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
	size_t height;
	size_t width;
	float* h_copy;
	float* d_copy{};
	c_matrix(size_t p_height, size_t p_width, float initial_max=0.5):
	height{p_height}, width{p_width}
	{
		size_t length = height * width;
		size_t float_size = sizeof(float);
		h_copy = new float[length];
		cudaMalloc((void**) &d_copy, float_size * length);
		fill_with_rand(h_copy, length, initial_max);
		for (int i = 0; i < height; ++i)
		{
			h_copy[i * width + width - 1] = 0;
		}
		cudaMemcpy(d_copy, h_copy, float_size * length, cudaMemcpyHostToDevice);
	}
	float* read()
	{
		cudaMemcpy(h_copy, d_copy, sizeof(float) * height * width, cudaMemcpyDeviceToHost);
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

__global__ void matTmulvec(float* mat, float* vec, int height, int width, float* out)
{
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	if (i < width)
	{
		float result = 0.0f;
		for (int j = 0; j < height; ++j)
		{
			result += mat[j * width + i] * vec[j];
		}
		out[i] = result;
	}
}

__global__ void relu_kernel(float* input, float* output, size_t size)
{
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	if (i < size)
	{
		if (input[i] < 0)
		{
			output[i] = 0;
		}else{
			output[i] = input[i];
		}
	}  
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

__global__ void elementwisemul(float* input1, float* input2, float* output, size_t size)
{
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	if (i < size)
	{
		output[i] = input1[i] * input2[i];
	}
}

__global__ void relu_derivative(float* input, float* output, size_t size)
{
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	if (i < size)
	{
		if (input[i] < 0)
		{
			output[i] = 0;
		}else{
			output[i] = 1;
		}
	}
}

__global__ void softmax_kernel(float* input, float* output, size_t size)
{
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	float sum = 0;
	for (int j = 0; j < size; ++j)
	{
		sum += expf(input[j]);
	}
	if (i < size)
	{
		output[i] = expf(input[i]) / sum;
	}
}

__global__ void softmax_crossen_error(float* input, float* output, size_t size, int target)
{
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	if (i < size)
	{
		if (i == target)
		{
			output[i] = (input[i] - 1);
		}else{
			output[i] = input[i];
		}

	}
}

__global__ void sigmoid_square_error(float* input, float* output, size_t size, int target)
{
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	if (i < size)
	{
		if (i == target)
		{
			output[i] = (input[i] - 1) * input[i] * (1 - input[i]);
		}else{
			output[i] = (input[i]) * input[i] * (1 - input[i]);
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

__global__ void weight_update_kernel(float* errors, float* last_activations, float* weights, float learning_rate)
{
	int i = threadIdx.x * gridDim.x + blockIdx.x;
	weights[i] += (-learning_rate * errors[threadIdx.x] * last_activations[blockIdx.x]);
}

class activation
{
public:
	void (*f)(float*, float*, size_t);
	void (*d)(float*, float*, size_t);
	activation(void (*p_f)(float*, float*, size_t), void (*p_d)(float*, float*, size_t)):
	f{p_f}, d{p_d}
	{} 
};

activation relu(relu_kernel, relu_derivative);
activation softmax(softmax_kernel, relu_derivative);
activation sigmoid(sigmoid_kernel, sigmoid_derivative);

class layer
{
public:
	size_t units;
	size_t input_length;
	c_vector activations;
	c_vector pre_activations;
	c_vector errors;
	c_matrix weights{1, 1};
	activation act;
	layer(size_t p_units=16, activation act_p=relu, size_t p_input_length=1):
	units{p_units}, activations{units + 1}, pre_activations{units}, errors{units},
	act{act_p}, input_length{p_input_length}
	{}
	void forward(float* input)
	{
		matmulvec<<<2, 8>>>(weights.d_copy, input, weights.height, weights.width, pre_activations.d_copy);
		act.f<<<2, 8>>>(pre_activations.d_copy, activations.d_copy, pre_activations.length);
	}
	void backward(c_matrix& nlw, c_vector& nle)
	{
		matTmulvec<<<2, 8>>>(nlw.d_copy, nle.d_copy, nlw.height, nlw.width, errors.d_copy);
		act.d<<<2, 8>>>(pre_activations.d_copy, pre_activations.d_copy, pre_activations.length);
		elementwisemul<<<2, 8>>>(errors.d_copy, pre_activations.d_copy, errors.d_copy, errors.length);
	}
	void set_input_lenght(size_t length)
	{
		input_length = length;
		weights = c_matrix(units, input_length);
	}

};

typedef void (*out_err_fptr)(float*, float*, size_t, int);

out_err_fptr get_out_err_func(
	void (*out_loss)(float*, float*, size_t),
	void (*out_act)(float*, float*, size_t))
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
		if (out_act == sigmoid_kernel)
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
	std::vector<layer> layers{};
	float* d_loss{};
	int* d_correct_label{};
	void (*loss_func)(float*, float*, size_t);
	void (*out_err_func)(float*, float*, size_t, int);
	bool final {false};
	float learning_rate;
	float* d_learning_rate;

	model(void (*p_loss_func)(float*, float*, size_t), float p_learning_rate):
	loss_func{p_loss_func}, learning_rate{p_learning_rate}
	{
		cudaMalloc((void**) &d_loss, sizeof(float));
		cudaMalloc((void**) &d_learning_rate, sizeof(float));
		cudaMalloc((void**) &d_correct_label, sizeof(int));
	}
	bool finalize()
	{
		if (get_out_err_func(loss_func, layers.back().act.f))
		{
			out_err_func = get_out_err_func(loss_func, layers.back().act.f);
			final = true;
			return true;
		}
		return false;
	}
	void add(layer l)
	{
		if(!layers.empty()){
			l.set_input_lenght(layers.back().activations.length);
			layers.push_back(l);
		}else{
			layers.push_back(l);
		}
	}
	void forward_pass(float* input_data)
	{
		cudaMemcpy(layers.front().activations.d_copy, input_data, sizeof(float) * layers.front().units, cudaMemcpyHostToDevice);
		float* temp_results = layers.front().activations.d_copy;
		for (std::vector<layer>::iterator l = layers.begin() + 1; l != layers.end(); ++l)
		{
			l->forward(temp_results);
			temp_results = l->activations.d_copy;
		}
	}
	void backprop(int target)
	{
		out_err_func<<<2, 8>>>(layers.back().activations.d_copy, layers.back().errors.d_copy, layers.back().units, target);
		for (std::vector<layer>::iterator l = layers.end() - 2; l != layers.begin(); --l)
		{
			l->backward((l + 1)->weights, (l + 1)->errors);
		}
	}
	void weight_update()
	{
		for (std::vector<layer>::iterator l = layers.begin() + 1; l != layers.end(); ++l)
		{
			weight_update_kernel<<<(l - 1)->activations.length, l->errors.length>>>
			(l->errors.d_copy, (l - 1)->activations.d_copy, l->weights.d_copy, learning_rate); 
		}
	}
	bool single_train(mnist_data_point& dp)
	{
		forward_pass(dp.image);
		float* result = layers.back().activations.read();
		int prediction = std::max_element(result, result + layers.back().units) - result;
		backprop(dp.label);
		weight_update();
		return prediction == dp.label;
	}
	bool single_test(mnist_data_point& dp)
	{
		forward_pass(dp.image);
		float* result = layers.back().activations.read();
		int prediction = std::max_element(result, result + layers.back().units) - result;
		return prediction == dp.label;
	}
	void train(std::vector<mnist_data_point>& data, int epochs)
	{
		if (finalize())
		{
			for (int epoch = 1; epoch <= epochs; ++epoch)
			{
				auto tik = std::chrono::high_resolution_clock::now();
				int num_of_data = data.size();
				float acc = 0;
				for (auto dp : data)
				{
					if(single_train(dp))
					{
						acc += 1.0f / num_of_data;
					}
				}
				auto tok = std::chrono::high_resolution_clock::now();
				std::chrono::duration<double, std::milli> ms_double = tok - tik;
				std::cout << "Epoch " << epoch << ": acc = " << acc << " in " << ms_double.count() << "ms.\n"; 
			}
		}else{
			std::cout << "Could not finalize model. \n";
		}
	}
	void test(std::vector<mnist_data_point>& data)
	{
		int num_of_data = data.size();
		float acc = 0;
		for (auto dp : data)
		{
			if(single_test(dp))
			{
				acc += 1.0f / num_of_data;
			}
		}
		std::cout << "test acc = " << acc << '\n';
	}
};

int main()
{
	std::srand(static_cast<unsigned int>(std::time(nullptr)));
	std::rand(); 

	auto test_data = mnist_parse("sample_data/mnist_test.csv");
	auto train_data = mnist_parse("sample_data/mnist_train_small.csv");

	// model mnist_model(mean_square_error, 0.5f);
	// mnist_model.add(layer(784));
	// mnist_model.add(layer(16, sigmoid));
	// mnist_model.add(layer(16, sigmoid));
	// mnist_model.add(layer(10, sigmoid));

	model mnist_model(cross_entropy, 0.01f);
	mnist_model.add(layer(784));
	mnist_model.add(layer(16));
	mnist_model.add(layer(16));
	mnist_model.add(layer(10, softmax));

	auto tik = std::chrono::high_resolution_clock::now();
	mnist_model.train(train_data, 3);
	auto tok = std::chrono::high_resolution_clock::now();
	std::chrono::duration<double, std::milli> ms_double = tok - tik;
	std::cout << ms_double.count() << "ms \n";
	// mnist_model.learning_rate = 0.005f;
	// mnist_model.train(train_data, 5);
	// mnist_model.learning_rate = 0.001f;
	// mnist_model.train(train_data, 5);


	mnist_model.test(test_data);

	return 0;
}

