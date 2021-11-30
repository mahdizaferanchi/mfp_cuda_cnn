Tensor sample_filter {3, 3, 1, 2};
class Identity : public Layer
{
public:
  size_t units;
  size_t input_length;
  bool double_activations;
  Identity(size_t p_units=16, Activation act_p=relu, bool p_double_activations=false, size_t p_input_length=1):
  Layer{act_p}, units{p_units}, input_length{p_input_length}, double_activations{p_double_activations}
  {}
  void forward(Tensor& input, cudaStream_t s)
  {
    // matmulmat<<<
    //   get_grids(input.height, units),
    //   get_threads(input.height, units),
    //   0, 
    //   s
    // >>>(input, weights, pre_activations);
    // act.f<<<
    //   get_grids(input.height, units),
    //   get_threads(input.height, units), 
    //   0, 
    //   s
    // >>>(pre_activations, activations);
    // cudaDeviceSynchronize();
    activations.write(input.read());
    cudaDeviceSynchronize();
  }
  void backward(Tensor& nlw, Tensor& nle, cudaStream_t s)
  {
    // matmulmatT<<<
    //   get_grids(nle.height, units), 
    //   get_threads(nle.height, units), 
    //   0, 
    //   s
    // >>>(nle, nlw, errors);
    // act.d<<<
    //   get_grids(pre_activations.height, units), 
    //   get_threads(pre_activations.height, units), 
    //   0, 
    //   s
    // >>>(pre_activations, pre_activations);
    // elementwisemul<<<
    //   get_grids(errors.height, units), 
    //   get_threads(errors.height, units), 
    //   0, 
    //   s
    // >>>(errors, pre_activations, errors);
    errors.write(nle.read());
    cudaDeviceSynchronize();
  }

  void update_weights(std::vector<std::reference_wrapper<Layer>>::iterator ll_iterator, float learning_rate, cudaStream_t stream, bool use_alt=false)
  {
    // weight_update_kernel<<<
    //   get_grids(weights.height, weights.width),
    //   get_threads(weights.height, weights.width),
    //   0, 
    //   stream
    // >>>(errors, use_alt ? (ll_iterator)->get().activations_alt : (ll_iterator)->get().activations, weights, learning_rate);
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
Tensor trans_filter {4, 4, 1, 2};
float vals[18] {2, 3, 4, 1, 0, 2, 1, 1, 1, 9, 0, 0, 0, 0.3, 0.1, 1, 1, 1};
sample_filter.write(vals);

Tensor G_matrix {4, 3};
float filter_transform_matrix_values[12] {1, 0, 0, 0.5, 0.5, 0.5 , 0.5, -0.5, 0.5, 0, 0, 1};
G_matrix.write(filter_transform_matrix_values);

flipped_filter_transform<<<
	1,
	dim3(4, 4, 2)
>>>(sample_filter, G_matrix, trans_filter);
std::cout << sample_filter << '\n';
std::cout << trans_filter << '\n';

// mnist_model.learning_rate = 0.001f;
// mnist_model.train(train_images, train_labels, 5, mini_batch_size);

// mnist_model.learning_rate = 0.0001f;
// mnist_model.train(train_images, train_labels, 5, mini_batch_size);

// mnist_model.reset_correct_predictions();
// mnist_model.single_test(test_images[0], test_labels[0], mini_batch_size);
// mnist_model.single_test(test_images[mini_batch_size], test_labels[mini_batch_size], mini_batch_size);
// // std::cout << mnist_model.layers[3].get().activations << '\n';
// float* results = mnist_model.layers[3].get().activations.read();
// for (int i = 0; i < 1 * mini_batch_size; ++i)
// {
// 	float max = 0;
// 	int maxIdx = 0;
// 	for (int j = 0; j < 10; ++j)
// 	{
// 		// std::cout << results[i* 11 + j] << ", ";
// 		if (max < results[i * 11 + j])
// 		{
// 			max = results[i * 11 + j];
// 			maxIdx = j;
// 			// std::cout << results[i * 11 + j] << ", ";
// 			// std::cout << max << ", ";
// 		}
// 	}
// 	std::cout << maxIdx << ", ";
// }
// std::cout << '\n';
// for (int i = mini_batch_size; i < 2 * mini_batch_size; ++i)
// {
// 	std::cout << *test_labels[i] << ", ";
// }
// std::cout << '\n';
// std::cout << "test acc = " << mnist_model.read_correct_predictions()/(float)mini_batch_size << '\n';
// std::cout << "test acc = " << mnist_model.read_correct_predictions() << '\n';

// for (int i = 0; i < 30; ++i)
// {
// 	mnist_model.forward_pass(train_data[i].image);
// 	std::cout << mnist_model.layers.back().activations;
// 	float* result = mnist_model.layers.back().activations.read();
// 	int prediction = std::max_element(result, result + mnist_model.layers.back().units) - result;
// 	std::cout << prediction << ' ';
// }
// model mnist_model(mean_square_error, 0.5f);
// mnist_model.add(layer(784));
// mnist_model.add(layer(16, sigmoid));
// mnist_model.add(layer(16, sigmoid));
// mnist_model.add(layer(10, sigmoid));

// mnist_model.move_batch(train_images[0], train_labels[0], 1, false);
// cudaDeviceSynchronize();
// mnist_model.forward_pass(1, false);
// mnist_model.layers[1].forward();
// cudaDeviceSynchronize();
// std::cout << mnist_model.layers[3].get().activations << '\n';
// std::cout << mnist_model.layers[3].get().errors << '\n';
// std::cout << mnist_model.layers[3].get().weights << '\n';
// std::cout << mnist_model.layers[2].get().activations << '\n';
// std::cout << mnist_model.layers[2].get().errors << '\n';
// std::cout << mnist_model.layers[2].get().weights << '\n';
// std::cout << mnist_model.layers[1].get().activations << '\n';
// std::cout << mnist_model.layers[1].get().errors << '\n';
// std::cout << mnist_model.layers[1].get().weights << '\n';
// std::cout << mnist_model.layers[0].get().activations << '\n';

// cudaDeviceProp props;
// cudaGetDeviceProperties(&props, 0);
// std::cout << props.sharedMemPerBlock << '\n';
// std::cout << cudaGetErrorName(cudaPeekAtLastError()) << '\n';

// __syncthreads();

// if (i < out.height && j < t_mat.height)
// {
// 	float result = 0.0f;
// 	for (int loopIdx = 0; loopIdx < out.width; ++loopIdx)
// 	{
// 		result += *out.at(i, loopIdx, k % in.depth, k / in.depth) * (*t_mat.at(j, loopIdx));
// 	}
// 	*out.at(i, j, k % in.depth, k / in.depth) = result;
// }

float bias_unit = 1.0;
constexpr int image_size = 784;
constexpr int hidden_1_size = 16;
constexpr int hidden_2_size = 16;
constexpr int output_size = 10;

Epoch 1: acc = 0.792175 in 1485.55ms.
Epoch 2: acc = 0.896543 in 1370.95ms.
Epoch 3: acc = 0.916646 in 1377.68ms.
4234.32ms 
test acc = 0.92014


float* hidden_1;
cudaMalloc((void**) &hidden_1, float_size * (hidden_1_size + 1));
cudaMemcpy(&hidden_1[hidden_1_size], &bias_unit, float_size, cudaMemcpyHostToDevice);

float* hidden_2;
cudaMalloc((void**) &hidden_2, float_size * (hidden_2_size + 1));
cudaMemcpy(&hidden_2[hidden_2_size], &bias_unit, float_size, cudaMemcpyHostToDevice);

float* output;
float h_output[output_size];
cudaMalloc((void**) &output, float_size * output_size);

float* loss;
cudaMalloc((void**) &loss, float_size);

int* correct_label;
cudaMalloc((void**) &correct_label, sizeof(int));

float h_weights_1[hidden_1_size][image_size + 1];
fill_with_rand(h_weights_1, hidden_1_size, image_size + 1);
float* weights_1;
cudaMalloc((void**) &weights_1, float_size * (image_size + 1) * hidden_1_size);
cudaMemcpy(weights_1, h_weights_1, float_size * (image_size + 1) * hidden_1_size, cudaMemcpyHostToDevice);

float h_weights_2[hidden_2_size][hidden_1_size + 1];
fill_with_rand(h_weights_2, hidden_2_size,  hidden_1_size + 1);
float* weights_2;
cudaMalloc((void**) &weights_2, float_size * (hidden_1_size + 1) * hidden_2_size);
cudaMemcpy(weights_2, h_weights_2, float_size * (hidden_1_size + 1) * hidden_2_size, cudaMemcpyHostToDevice);

float h_weights_o[output_size][hidden_2_size + 1];
fill_with_rand(h_weights_o, output_size, hidden_2_size + 1);
float* weights_o;
cudaMalloc((void**) &weights_o, float_size * (hidden_2_size + 1) * output_size);
cudaMemcpy(weights_o, h_weights_o, float_size * (hidden_2_size + 1) * output_size, cudaMemcpyHostToDevice);

float* image;
cudaMalloc((void **) &image, float_size * image_size);
cudaMemcpy(image, &train_data[0].image, float_size * image_size, cudaMemcpyHostToDevice);
cudaMemcpy(correct_label, &train_data[0].label, sizeof(int), cudaMemcpyHostToDevice);

//forward pass:
matmulvecrelu<<<2, 8>>>(weights_1, image, hidden_1_size, image_size + 1, hidden_1);
matmulvecrelu<<<2, 8>>>(weights_2, hidden_1, hidden_2_size, hidden_1_size + 1, hidden_2);
matmulvec<<<2, 5>>>(weights_o, hidden_2, output_size, hidden_2_size + 1, output);
softmax<<<1, 10>>>(output, output_size);

//loss calculation:
calc_loss<<<1, 10>>>();

cudaMemcpy(h_output, output, float_size * output_size, cudaMemcpyDeviceToHost);

for (int i = 0; i < output_size; ++i)
{
	std::cout << h_output[i] << " ";
}
std::cout << '\n';

import tensorflow as tf
from tensorflow.keras.layers import Flatten
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.losses import SparseCategoricalCrossentropy
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import MaxPooling2D
import numpy as np

(x_train, y_train), (x_test, y_test) = tf.keras.datasets.fashion_mnist.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0
x_test = x_test.reshape(10000, 28, 28, 1)
x_train = x_train.reshape(60000, 28, 28, 1)

model = Sequential()
model.add(Flatten())
model.add(Dense(16, activation='relu'))
model.add(Dense(16, activation='relu'))
model.add(Dense(10, activation='softmax'))
model.compile('sgd', loss=SparseCategoricalCrossentropy(), metrics=['accuracy'])
useless = model(x_train[0:1])
# model.fit(x_train, y_train, batch_size=32, epochs=1, verbose=2)
# model.evaluate(x_test, y_test, batch_size=32, verbose=2)

# model.get_layer(index=1).set_weights(w_1)
# model.get_layer(index=2).set_weights(w_2)
# model.get_layer(index=3).set_weights(w_3)
print(model.get_layer(index=2).get_weights())

# model.fit(x_train[0:20000], y_train[0:20000], batch_size=1, epochs=1, verbose=2)
# model.evaluate(x_test, y_test, batch_size=32, verbose=2)

# print(y_train[0:30])
pre = model(x_train[0:30])
ans = [np.argmax(ten) for ten in pre]
print(ans)

thefile = open("sample_data/mnist_train_small.csv")
thefile.seek(2)
# arr = []
# thefile.read(1)
# thefile.read(1)
# for i in range(10):
#   arr.append(int(thefile.read(1)))
#   thefile.read(1)

# npthing = np.array(arr)
# print(npthing)

target = [int(i)/255.0 for i in thefile.readline().split(',')]
ex = np.array(target, dtype="float32").reshape(1, 28, 28, 1)

// std::cout << test_data[0].label << '\n';
// int idx = 0;
// while(!test_data[0].image[idx]) {++idx;}
// std::cout << test_data[0].image[idx] << '\n';

// int t = 7;
// mnist_model.forward_pass(train_data[t].image);
// float* result = mnist_model.layers.back().activations.read();
// std::cout << mnist_model.layers.back().weights << '\n';
// std::cout << mnist_model.layers.back().activations << '\n';
// int prediction = std::max_element(result, result + mnist_model.layers.back().units) - result;
// std::cout << train_data[t].label << '\n';
// mnist_model.backprop(train_data[t].label);
// mnist_model.weight_update();
// mnist_model.forward_pass(train_data[t].image);
// mnist_model.backprop(train_data[t].label);
// mnist_model.weight_update();
// mnist_model.forward_pass(train_data[t].image);
// mnist_model.backprop(train_data[t].label);
// mnist_model.weight_update();
// mnist_model.forward_pass(train_data[t].image);
// mnist_model.backprop(train_data[t].label);
// mnist_model.weight_update();
// mnist_model.forward_pass(train_data[t].image);
// mnist_model.backprop(train_data[t].label);
// mnist_model.weight_update();

// std::cout << mnist_model.layers.back().weights << '\n';
// std::cout << mnist_model.layers.back().activations;

// model small_test(cross_entropy, 0.1f);
// small_test.add(layer(2));
// small_test.add(layer(2));
// small_test.add(layer(4, softmax));

// float test_input[2]{1, 1.2};
// float test_input2[2]{0.5, 1.1};

// small_test.forward_pass(test_input);

// std::cout << small_test.layers[0].activations << '\n';
// std::cout << small_test.layers[1].weights << '\n';
// std::cout << small_test.layers[1].activations << '\n';
// std::cout << small_test.layers[2].weights << '\n';
// std::cout << small_test.layers[2].activations << '\n';

// std::cout << "update" << '\n';

// int test = 0;

// small_test.backprop(test);
// small_test.weight_update();
// small_test.forward_pass(test_input);

// std::cout << "**" << '\n' << small_test.layers[1].errors << '\n';

// std::cout << small_test.layers[0].activations << '\n';
// std::cout << small_test.layers[1].weights << '\n';
// std::cout << small_test.layers[1].activations << '\n';
// std::cout << small_test.layers[2].weights << '\n';
// std::cout << small_test.layers[2].activations << '\n';
// std::cout << small_test.layers[2].pre_activations << '\n';

import numpy as np

G = np.array([[1, 0, 0], 
              [0.5, 0.5, 0.5],
              [0.5, -0.5, 0.5],
              [0, 0, 1]])

B = np.array([[1, 0, -1, 0],
              [0, 1 , 1, 0],
              [0, -1, 1, 0],
              [0, 1, 0, -1]]);

input = np.array([[-0.218471,   0.265666,   0.124059,    0.38729],
                  [0.123174,  -0.224023,   0.413626,   0.404703],
                  [-0.286517,   0.296222,   0.143381,   0.336657],
                  [0.388757,  0.0698263,  0.0542908,  -0.397761]])
filter = np.array([[0.150511,  -0.286974,   0.241168],
                   [-0.455907,  -0.381255,   0.262464],
                   [-0.0162981,   0.239053,    0.11768]])

input_inter = np.matmul(B, input)
filter_inter = np.matmul(G, filter)

input_transform = np.matmul(input_inter, np.transpose(B))
filter_transform = np.matmul(filter_inter, np.transpose(G))

output = np.multiply(input_transform, filter_transform)
print(output)


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
			// *inter.at((2 * yIdx + vIdx), (2 * xIdx + hIdx), (k % in.depth), (k / in.depth)) = result;
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
				// result += (*inter.at(2 * yIdx + vIdx, 2 * xIdx + loopIdx)) * (*t_mat.at(hIdx, loopIdx));
				result += intermediate[vIdx][loopIdx] * (*t_mat.at(hIdx, loopIdx));
			}
			*out.at((2 * yIdx + vIdx), (2 * xIdx + hIdx), (k % in.depth), (k / in.depth)) = result;
		}
	}	
}

 //*******************<new test>*********************
  Tensor cbt_acts {6, 6, 2, 2};
  // Tensor cbt_acts {6, 6};
  float cbt_acts_vals[36 * 4] {0, 0, 0, 0, 0, 0,0, 0, 0, 0, 0, 0,0, 0, 0, 0, 0, 0,0, 0, 0, 0, 0, 0,0, 0, 0, 0, 0, 0,0, 0, 0, 0, 0, 0,0, 0, 0, 0, 0, 0,0, 0, 0, 0, 0, 0,0, 0, 0, 0, 0, 0,0, 0, 0, 0, 0, 0,0, 0, 0, 0, 0, 0,0, 0, 0, 0, 0, 0,0, 0, 0, 0, 0, 0,0, 0, 0, 0, 0, 0,0, 0, 0, 0, 0, 0,0, 0, 0, 0, 0, 0,0, 0, 0, 0, 0, 0,0, 0, 0, 0, 0, 0,1.0, 0.6, -0.5, 0.0, 1.0, 2.0, 2.0, 1.0, 1.0, 0.0, 0.0, 0.0, -1.0, -1.0, -1.0, -1.0, -1.0, 0.3, 0.8, -0.8, 0.2, 0.2, 1.0, 1.0, 2.0, 2.0, -3.0, 0.0, -0.5, 5.0, 0.8, -0.8, 0.2, 0.2, 1.0, 1.0};
  // float cbt_acts_vals[36] {1.0, 0.6, -0.5, 0.0, 1.0, 2.0, 2.0, 1.0, 1.0, 0.0, 0.0, 0.0, -1.0, -1.0, -1.0, -1.0, -1.0, 0.3, 0.8, -0.8, 0.2, 0.2, 1.0, 1.0, 2.0, 2.0, -3.0, 0.0, -0.5, 5.0, 0.8, -0.8, 0.2, 0.2, 1.0, 1.0};
  // float cbt_acts_vals[36] {1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1};
  cbt_acts.write(cbt_acts_vals);
  Tensor cbt_errs {6, 6, 2, 2};
  // Tensor cbt_errs {6, 6};
  // float cbt_errs_vals[36] {1,1,1,1,1,1,1,1, 1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1};
  float cbt_errs_vals[36 * 4] {0, 0, 0, 0, 0, 0,0, 0, 0, 0, 0, 0,0, 0, 0, 0, 0, 0,0, 0, 0, 0, 0, 0,0, 0, 0, 0, 0, 0,0, 0, 0, 0, 0, 0,0, 0, 0, 0, 0, 0,0, 0, 0, 0, 0, 0,0, 0, 0, 0, 0, 0,0, 0, 0, 0, 0, 0,0, 0, 0, 0, 0, 0,0, 0, 0, 0, 0, 0,0, 0, 0, 0, 0, 0,0, 0, 0, 0, 0, 0,0, 0, 0, 0, 0, 0,0, 0, 0, 0, 0, 0,0, 0, 0, 0, 0, 0,0, 0, 0, 0, 0, 0,0.5, 0.1, -1.0, 1.0, 0.3, 2.0, 0.9, 0.2, -1.0, 1.0, 0.4, 0.5, 1.0, 0.0, -1.0, 2.0, 1.0, 0.01, 1.0, 0.0, -1.0, 98.0, 1.0, 1.0, 1.0, 0.0, -1.0, 1.0, 0.4, 0.5, 1.0, 0.0, -1.0, 1.0, 0.4, 0.5};
  // float cbt_errs_vals[36] {0.5, 0.1, -1.0, 1.0, 0.3, 2.0, 0.9, 0.2, -1.0, 1.0, 0.4, 0.5, 1.0, 0.0, -1.0, 2.0, 1.0, 0.01, 1.0, 0.0, -1.0, 98.0, 1.0, 1.0, 1.0, 0.0, -1.0, 1.0, 0.4, 0.5, 1.0, 0.0, -1.0, 1.0, 0.4, 0.5};
  cbt_errs.write(cbt_errs_vals);
  Tensor result {3, 3, 2, 2};
  // Tensor result {3, 3};
  // transform ll activations
  Tensor ll_trans_acts {
    cbt_acts.height * 2,
    cbt_acts.width * 2,
    cbt_acts.depth,
    cbt_acts.fourth
  };
  map_transform<<<
    dim3(1, 1, cbt_acts.depth * cbt_acts.fourth),
    dim3(cbt_acts.height / 2, cbt_acts.width / 2)
  >>>(cbt_acts, layer1.B2_matrix, ll_trans_acts);
  cudaDeviceSynchronize();

  // transform errors
  Tensor transformed_errors {cbt_errs.height * 2, cbt_errs.width * 2, cbt_errs.depth, cbt_errs.fourth};
  map_transform_for_backprop<<<
    dim3(1, 1, cbt_errs.depth * cbt_errs.fourth),
    dim3(cbt_errs.height / 2, cbt_errs.width / 2)
  >>>(cbt_errs, layer1.G2_matrix, transformed_errors);
  cudaDeviceSynchronize();

  // mul in wts with R * S reslut
  Tensor conv_ans {4, 4, result.depth, result.fourth, true, 0};
  wts_ll_acts_mul_errs<<<
    dim3(1, 1, ll_trans_acts.fourth * transformed_errors.depth * ll_trans_acts.depth),
    dim3(ll_trans_acts.height / 4, ll_trans_acts.width / 4)
  >>>(ll_trans_acts, transformed_errors, conv_ans);
  cudaDeviceSynchronize();

  // inverse transform
  inverse_transform_for_weights<<<
    dim3(1, 1, conv_ans.depth * conv_ans.fourth),
    dim3(conv_ans.height / 4, conv_ans.width / 4)
  >>>(conv_ans, layer1.A2_matrix, 1.0, result);

  std::cout << conv_ans << '\n';
  std::cout << ll_trans_acts << '\n';
  std::cout << transformed_errors << '\n';
  std::cout << result << '\n';
  //*******************</new test>********************* 
// for (std::vector<std::reference_wrapper<Layer>>::iterator l = mnist_model.layers.begin() + 1; l != mnist_model.layers.end(); ++l)
// {
//   std::cout << l->get().weights << '\n';
// }

// std::cout << mnist_model.layers[2].get().activations << '\n';
// std::cout << mnist_model.layers[3].get().errors << '\n';
// std::cout << mnist_model.layers[3].get().weights << '\n';

// Tensor fake_nle {3, 3, 2, mini_batch_size};
// Tensor fake_nlw {3, 3, 2, mini_batch_size};

// mnist_model.layers[1].get().backward(
//   mnist_model.layers[2].get().weights,
//   mnist_model.layers[2].get().errors,
//   mnist_model.kernel_exec_s
// );

// mnist_model.layers[2].get().backward(
//   mnist_model.layers[3].get().weights,
//   mnist_model.layers[3].get().errors,
//   mnist_model.kernel_exec_s
// );

// std::cout << cudaGetErrorName(cudaPeekAtLastError()) << '\n';

// std::cout << "Input : \n";
// std::cout << input << '\n';

// std::cout << "**** Transformed Input:\n";
// std::cout << transformed_input << '\n';
// std::cout << "***\n";

// std::cout << "Weights : \n";
// std::cout << weights;

// std::cout << "**** Transformed Weights:\n";
// std::cout << transformed_weights << '\n';
// std::cout << "***\n";

// std::cout << "after mul : \n";
// std::cout << conv_ans << '\n';

// std::cout << "final result: \n";
// std::cout << activations << '\n';
  Convolutional& conv_layer = dynamic_cast<Convolutional&>(mnist_model.layers[1].get());
  std::cout << conv_layer.biases << '\n';
*** 5 is the cut off point *** (layer3 units : 128)
  // cudaDeviceSynchronize();
  // mnist_model.single_train(train_images[0], train_labels[0], mini_batch_size);
  // cudaDeviceSynchronize();


  // mnist_model.layers[1].get().weights.make_file("l2_weights.t");
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
  // layer2.ll_transformed_acts.make_file("transformed_l1_acts.t");
  // layer2.transformed_errors.make_file("transformed_l2_errors.t");
  // layer2.weight_update_inter.make_file("l2_weight_update_inter.t");
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

  // mnist_model.move_batch(train_images[0], train_labels[0], mini_batch_size, false);
  // cudaDeviceSynchronize();
  // std::cout << cudaGetErrorName(cudaPeekAtLastError()) << '\n';
  // mnist_model.forward_pass(mini_batch_size, false);
  // cudaDeviceSynchronize();
  // std::cout << cudaGetErrorName(cudaPeekAtLastError()) << '\n';
  // mnist_model.layers[2].get().weights.make_file("l3_weights.t");
  // layer2.pre_activations.make_file("l2_pre_activations_before.t");
  // mnist_model.backprop(mini_batch_size, false);
  // cudaDeviceSynchronize();
  // std::cout << cudaGetErrorName(cudaPeekAtLastError()) << '\n';
  // layer2.transformed_flipped_weights.make_file("l2_transformed_flipped_weights.t");
  // layer2.transfromed_nle.make_file("l2_transfromed_nle.t");
  // layer2.backward_conv_inter.make_file("l2_backward_conv_inter.t");
  // layer2.errors.make_file("l2_errors.t");
  // layer2.pre_activations.make_file("l2_pre_activations.t");
  // mnist_model.weight_update(false);
  // cudaDeviceSynchronize();
  // std::cout << cudaGetErrorName(cudaPeekAtLastError()) << '\n';

