import { AdamW } from "./AdamW.js";
import { randomNormal } from "./Random.js";

/**
 * The activation functions a hidden layer can use:
 *   "relu"    - max(0, x). The usual choice for deep networks.
 *   "sigmoid" - 1 / (1 + e^-x), which squashes every value into 0-1. Its slope
 *               is never exactly 0, so a neuron can always recover, which
 *               helps very small networks such as XORNetwork.
 */
export type Activation = "relu" | "sigmoid";

/**
 * A fully connected hidden layer: every neuron receives every input.
 * Hidden layers use the ReLU activation function unless told otherwise;
 * the weight initialization in build() is chosen to match the activation.
 */
export class HiddenLayer {
    private readonly _inputSize: number;
    private readonly _numNeurons: number;
    private readonly _activation: Activation;
    private _weights: number[][] | undefined;
    private _biases: number[] | undefined;

    // From the most recent call to update(), kept for backpropagation.
    private _inputs: number[] | undefined;
    private _activations: number[] | undefined;

    // From the most recent call to calculateErrors().
    private _errors: number[] | undefined;

    // From the most recent call to calculateGradients().
    private _weightGradients: number[][] | undefined;
    private _biasGradients: number[] | undefined;
    private _gradientCount = 0;

    /**
     * @param inputSize Number of values coming into the layer, e.g. 64 for a flattened 8x8 image.
     * @param numNeurons Number of neurons in the layer, which is also the number of values it outputs.
     * @param activation The activation function applied by every neuron.
     */
    constructor(inputSize: number, numNeurons: number, activation: Activation = "relu") {
        if (!Number.isInteger(inputSize) || inputSize < 1) {
            throw new RangeError(`inputSize must be a positive integer, got ${inputSize}`);
        }
        if (!Number.isInteger(numNeurons) || numNeurons < 1) {
            throw new RangeError(`numNeurons must be a positive integer, got ${numNeurons}`);
        }
        this._inputSize = inputSize;
        this._numNeurons = numNeurons;
        this._activation = activation;
    }

    /**
     * Creates the weight matrix and bias vector and sets their starting values.
     * Calling it again discards the current weights and biases and starts afresh.
     *
     * The matrix has one row per neuron and one column per input, so
     * weights[n][i] is the weight neuron n gives to input i.
     *
     * Uses He initialization: values are drawn from a normal distribution with
     * mean 0 and standard deviation sqrt(2 / inputSize). This keeps the size of
     * the signal roughly constant as it passes through ReLU layers. Equal or
     * zero weights would not work, because every neuron would then compute the
     * same thing and learn the same thing.
     *
     * For sigmoid neurons, Xavier (Glorot) initialization is used instead:
     * standard deviation sqrt(2 / (inputSize + numNeurons)). Sigmoid passes on
     * about a quarter of the slope that ReLU does, and He's larger weights
     * would push it into its flat regions, where it learns slowly.
     *
     * The MNIST pixels are normalized to 0-1 by ImageLoader, which is the input
     * scale this initialization assumes.
     *
     * There is one bias per neuron, so biases[n] is added to neuron n's weighted
     * sum. Biases start at 0: the random weights already make the neurons
     * different, so the biases do not need to be random.
     */
    build(): void {
        const standardDeviation = this._activation === "relu"
            ? Math.sqrt(2 / this._inputSize)
            : Math.sqrt(2 / (this._inputSize + this._numNeurons));
        this._weights = [];
        for (let n = 0; n < this._numNeurons; n++) {
            const row: number[] = [];
            for (let i = 0; i < this._inputSize; i++) {
                row.push(randomNormal() * standardDeviation);
            }
            this._weights.push(row);
        }
        this._biases = new Array<number>(this._numNeurons).fill(0);
    }

    /**
     * Feeds a set of inputs through the layer and returns each neuron's output.
     * For each neuron, every input is multiplied by its weight, the products
     * are summed, the bias is added and the activation function (normally
     * ReLU) is applied.
     *
     * The inputs and the outputs (activations) are stored, replacing those from
     * the previous call. Backpropagation needs them to calculate this layer's
     * gradients.
     *
     * @param inputs Values coming into the layer; must have inputSize elements.
     * @returns One output per neuron.
     * @throws RangeError if the number of inputs does not match inputSize.
     * @throws Error if build() has not been called yet.
     */
    update(inputs: readonly number[]): number[] {
        if (inputs.length !== this._inputSize) {
            throw new RangeError(`Expected ${this._inputSize} inputs, got ${inputs.length}`);
        }
        const weights = this.weights;
        const biases = this.biases;

        const outputs: number[] = [];
        for (let n = 0; n < this._numNeurons; n++) {
            const neuronWeights = weights[n]!;
            let sum = 0;
            for (let i = 0; i < this._inputSize; i++) {
                sum += inputs[i]! * neuronWeights[i]!;
            }
            sum += biases[n]!;
            outputs.push(this._activation === "relu" ? relu(sum) : sigmoid(sum));
        }

        // Store copies, so changes the caller makes to either array do not alter them.
        this._inputs = [...inputs];
        this._activations = outputs;
        return [...outputs];
    }

    /**
     * Calculates each neuron's error for the most recent call to update(),
     * from the errors passed back by the next layer:
     *
     *     error[n] = errorFromNextLayer[n] * ReLU slope[n]
     *
     * The ReLU slope is 1 if the neuron's activation was above 0, and 0 if it
     * was 0. A neuron that output 0 had no effect on the next layer for this
     * image, so its error is 0. (ReLU's output is above 0 exactly when its
     * input was, so the stored activations tell us the slope.)
     *
     * For sigmoid neurons the slope is activation * (1 - activation), which
     * is largest (0.25) when the activation is 0.5 and approaches 0, without
     * reaching it, as the activation approaches 0 or 1.
     *
     * The errors are also stored, replacing those from the previous call, and
     * can be read through the errors property.
     *
     * @param errorsFromNextLayer One value per neuron, from the next layer's calculatePreviousLayerErrors().
     * @returns One error per neuron, which can be passed to calculateGradients().
     * @throws RangeError if there is not one value per neuron.
     * @throws Error if update() has not been called yet.
     */
    calculateErrors(errorsFromNextLayer: readonly number[]): number[] {
        if (errorsFromNextLayer.length !== this._numNeurons) {
            throw new RangeError(`Expected ${this._numNeurons} errors (one per neuron), got ${errorsFromNextLayer.length}`);
        }
        const activations = this.activations;

        const errors: number[] = [];
        for (let n = 0; n < this._numNeurons; n++) {
            const activation = activations[n]!;
            const slope = this._activation === "relu"
                ? (activation > 0 ? 1 : 0)
                : activation * (1 - activation);
            errors.push(errorsFromNextLayer[n]! * slope);
        }

        // Store the errors, and return a copy so changes the caller makes do not alter them.
        this._errors = errors;
        return [...errors];
    }

    /**
     * Passes this layer's errors back through its weights, giving the error
     * for each of its inputs, i.e. for each neuron in the previous layer:
     *
     *     previousLayerError[i] = sum over n of ( weight[n][i] * error[n] )
     *
     * Input i feeds every neuron n in this layer through weight[n][i], so its
     * share of the blame is each neuron's error weighted by that connection.
     * (In matrix terms: the transposed weight matrix times the error vector.)
     *
     * If the previous layer is a hidden layer, it must still multiply these
     * values by its own ReLU slope; see calculateErrors(). For the first hidden
     * layer the inputs are the image's pixels, which have no weights to learn,
     * so there is no need to call this method on it.
     *
     * Call this before the weights are updated, so it uses the weights that
     * produced the current output.
     *
     * @param errors This layer's errors, one per neuron, from calculateErrors().
     * @returns One value per input (inputSize values), to pass to the previous layer.
     * @throws RangeError if there is not one error per neuron.
     * @throws Error if build() has not been called yet.
     */
    calculatePreviousLayerErrors(errors: readonly number[]): number[] {
        if (errors.length !== this._numNeurons) {
            throw new RangeError(`Expected ${this._numNeurons} errors (one per neuron), got ${errors.length}`);
        }
        const weights = this.weights;

        const previousLayerErrors: number[] = [];
        for (let i = 0; i < this._inputSize; i++) {
            let sum = 0;
            for (let n = 0; n < this._numNeurons; n++) {
                sum += weights[n]![i]! * errors[n]!;
            }
            previousLayerErrors.push(sum);
        }
        return previousLayerErrors;
    }

    /**
     * Calculates the gradients for the most recent inputs passed to update()
     * and stores them in weightGradients and biasGradients.
     *
     * Each neuron n works out a weighted sum before applying ReLU:
     *
     *     sum[n] = sum over i of ( weight[n][i] * input[i] ) + bias[n]
     *
     * The error for neuron n (from calculateErrors(), which has already taken
     * the ReLU slope into account) tells us how much the loss changes when
     * sum[n] changes. Using the chain rule:
     *
     *     weight gradient[n][i] = error[n] * input[i]
     *     bias gradient[n]      = error[n]
     *
     * This is the same rule as in the output layer. A neuron whose output was
     * 0 has an error of 0, so all of its gradients are 0 for this image.
     *
     * Averaging over a batch
     * ----------------------
     * When batchAverage is true, the new gradients are combined with the
     * stored ones using a running mean, so after k images the stored
     * gradients are the average of those k images' gradients:
     *
     *     mean = mean + (new - mean) / k
     *
     * This gives the same result as adding up all k gradients and dividing
     * by k, without having to keep every image's gradients. When batchAverage
     * is false, the new gradients replace the stored ones, starting a new
     * average (k = 1). Use false for the first image of each batch and true
     * for the rest.
     *
     * @param errors One error per neuron, from calculateErrors().
     * @param batchAverage true to average the new gradients with the stored
     * ones; false to replace them.
     * @throws RangeError if there is not one error per neuron.
     * @throws Error if update() has not been called yet.
     */
    calculateGradients(errors: readonly number[], batchAverage: boolean): void {
        if (errors.length !== this._numNeurons) {
            throw new RangeError(`Expected ${this._numNeurons} errors (one per neuron), got ${errors.length}`);
        }
        const inputs = this.inputs;

        // Step 1: weight gradients, one per weight.
        const weightGradients: number[][] = [];
        for (let n = 0; n < this._numNeurons; n++) {
            const neuronWeightGradients: number[] = [];
            for (let i = 0; i < this._inputSize; i++) {
                neuronWeightGradients.push(errors[n]! * inputs[i]!);
            }
            weightGradients.push(neuronWeightGradients);
        }

        // Step 2: bias gradients, one per neuron. A bias is added directly to
        // the sum (as if its input were always 1), so its gradient is just
        // the error.
        const biasGradients: number[] = [];
        for (let n = 0; n < this._numNeurons; n++) {
            biasGradients.push(errors[n]!);
        }

        // Step 3: store the gradients. Replace the stored ones, unless we are
        // averaging and there are stored gradients to average with.
        if (!batchAverage || this._weightGradients === undefined || this._biasGradients === undefined) {
            this._weightGradients = weightGradients;
            this._biasGradients = biasGradients;
            this._gradientCount = 1;
            return;
        }
        this._gradientCount++;
        const k = this._gradientCount;
        for (let n = 0; n < this._numNeurons; n++) {
            const meanWeightGradients = this._weightGradients[n]!;
            for (let i = 0; i < this._inputSize; i++) {
                meanWeightGradients[i]! += (weightGradients[n]![i]! - meanWeightGradients[i]!) / k;
            }
            this._biasGradients[n]! += (biasGradients[n]! - this._biasGradients[n]!) / k;
        }
    }

    /**
     * Updates this layer's weights in place using the optimizer and the
     * stored (batch-averaged) weight gradients. Weight decay is applied.
     *
     * Call optimizer.step() once per batch before updating any layer, and
     * call this only after backpropagation has used the current weights
     * (calculatePreviousLayerErrors() reads them).
     *
     * @param optimizer The AdamW optimizer, shared by every layer in the network.
     * @throws Error if build() or calculateGradients() has not been called yet.
     */
    updateWeights(optimizer: AdamW): void {
        if (this._weights === undefined) {
            throw new Error("HiddenLayer weights have not been created yet; call build() first");
        }
        optimizer.updateMatrix(this._weights, this.weightGradients);
    }

    /**
     * Updates this layer's biases in place using the optimizer and the stored
     * (batch-averaged) bias gradients. No weight decay is applied to biases.
     *
     * @param optimizer The AdamW optimizer, shared by every layer in the network.
     * @throws Error if build() or calculateGradients() has not been called yet.
     */
    updateBiases(optimizer: AdamW): void {
        if (this._biases === undefined) {
            throw new Error("HiddenLayer biases have not been created yet; call build() first");
        }
        optimizer.updateVector(this._biases, this.biasGradients);
    }
    /**
     * Returns the number of values in the layer that training can adjust:
     * one weight per input for each neuron, plus one bias per neuron.
     * This depends only on the layer's size, so build() does not need to have been called.
     */
    getParameterCount(): number {
        return this._numNeurons * this._inputSize + this._numNeurons;
    }

    /** Number of values coming into the layer. */
    get inputSize(): number {
        return this._inputSize;
    }

    /** Number of neurons in the layer. */
    get numNeurons(): number {
        return this._numNeurons;
    }

    /** The activation function applied by every neuron. */
    get activation(): Activation {
        return this._activation;
    }

    /**
     * The weight matrix, indexed [neuron][input].
     * @throws Error if build() has not been called yet.
     */
    get weights(): readonly (readonly number[])[] {
        if (this._weights === undefined) {
            throw new Error("HiddenLayer weights have not been created yet; call build() first");
        }
        return this._weights;
    }

    /**
     * The bias vector, one value per neuron.
     * @throws Error if build() has not been called yet.
     */
    get biases(): readonly number[] {
        if (this._biases === undefined) {
            throw new Error("HiddenLayer biases have not been created yet; call build() first");
        }
        return this._biases;
    }

    /**
     * The inputs from the most recent update() call.
     * @throws Error if update() has not been called yet.
     */
    get inputs(): readonly number[] {
        if (this._inputs === undefined) {
            throw new Error("HiddenLayer has no stored inputs yet; call update() first");
        }
        return this._inputs;
    }

    /**
     * Each neuron's output (after ReLU) from the most recent update() call.
     * @throws Error if update() has not been called yet.
     */
    get activations(): readonly number[] {
        if (this._activations === undefined) {
            throw new Error("HiddenLayer has no stored activations yet; call update() first");
        }
        return this._activations;
    }

    /**
     * Each neuron's error from the most recent calculateErrors() call.
     * @throws Error if calculateErrors() has not been called yet.
     */
    get errors(): readonly number[] {
        if (this._errors === undefined) {
            throw new Error("HiddenLayer has no stored errors yet; call calculateErrors() first");
        }
        return this._errors;
    }

    /**
     * The weight gradients from the most recent calculateGradients() call,
     * indexed [neuron][input] like the weight matrix.
     * @throws Error if calculateGradients() has not been called yet.
     */
    get weightGradients(): readonly (readonly number[])[] {
        if (this._weightGradients === undefined) {
            throw new Error("HiddenLayer has no weight gradients yet; call calculateGradients() first");
        }
        return this._weightGradients;
    }

    /**
     * The bias gradients from the most recent calculateGradients() call, one per neuron.
     * @throws Error if calculateGradients() has not been called yet.
     */
    get biasGradients(): readonly number[] {
        if (this._biasGradients === undefined) {
            throw new Error("HiddenLayer has no bias gradients yet; call calculateGradients() first");
        }
        return this._biasGradients;
    }

    /** Number of images averaged into the stored gradients (0 before calculateGradients() is called). */
    get gradientCount(): number {
        return this._gradientCount;
    }
}

/**
 * Rectified Linear Unit: passes positive values through unchanged and
 * turns negative values into 0.
 */
function relu(x: number): number {
    return Math.max(0, x);
}

/**
 * The logistic sigmoid: squashes any value into the range 0-1, giving 0.5
 * for 0, nearly 1 for large positive values and nearly 0 for large negative ones.
 */
function sigmoid(x: number): number {
    return 1 / (1 + Math.exp(-x));
}
