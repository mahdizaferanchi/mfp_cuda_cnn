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