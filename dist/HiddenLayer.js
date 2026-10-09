import { randomNormal } from "./Random.js";
/**
 * A fully connected hidden layer: every neuron receives every input.
 * Hidden layers are assumed to use the ReLU activation function, which is
 * what the weight initialization in build() is chosen for.
 */
export class HiddenLayer {
    _inputSize;
    _numNeurons;
    _weights;
    _biases;
    // From the most recent call to calculate(), kept for backpropagation.
    _inputs;
    _activations;
    /**
     * @param inputSize Number of values coming into the layer, e.g. 64 for a flattened 8x8 image.
     * @param numNeurons Number of neurons in the layer, which is also the number of values it outputs.
     */
    constructor(inputSize, numNeurons) {
        if (!Number.isInteger(inputSize) || inputSize < 1) {
            throw new RangeError(`inputSize must be a positive integer, got ${inputSize}`);
        }
        if (!Number.isInteger(numNeurons) || numNeurons < 1) {
            throw new RangeError(`numNeurons must be a positive integer, got ${numNeurons}`);
        }
        this._inputSize = inputSize;
        this._numNeurons = numNeurons;
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
     * The MNIST pixels are normalized to 0-1 by ImageLoader, which is the input
     * scale this initialization assumes.
     *
     * There is one bias per neuron, so biases[n] is added to neuron n's weighted
     * sum. Biases start at 0: the random weights already make the neurons
     * different, so the biases do not need to be random.
     */
    build() {
        const standardDeviation = Math.sqrt(2 / this._inputSize);
        this._weights = [];
        for (let n = 0; n < this._numNeurons; n++) {
            const row = [];
            for (let i = 0; i < this._inputSize; i++) {
                row.push(randomNormal() * standardDeviation);
            }
            this._weights.push(row);
        }
        this._biases = new Array(this._numNeurons).fill(0);
    }
    /**
     * Feeds a set of inputs through the layer and returns each neuron's output.
     * For each neuron, every input is multiplied by its weight, the products
     * are summed, the bias is added and the ReLU function is applied.
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
    calculate(inputs) {
        if (inputs.length !== this._inputSize) {
            throw new RangeError(`Expected ${this._inputSize} inputs, got ${inputs.length}`);
        }
        const weights = this.weights;
        const biases = this.biases;
        const outputs = [];
        for (let n = 0; n < this._numNeurons; n++) {
            const neuronWeights = weights[n];
            let sum = 0;
            for (let i = 0; i < this._inputSize; i++) {
                sum += inputs[i] * neuronWeights[i];
            }
            sum += biases[n];
            outputs.push(relu(sum));
        }
        // Store copies, so changes the caller makes to either array do not alter them.
        this._inputs = [...inputs];
        this._activations = outputs;
        return [...outputs];
    }
    /**
     * Returns the number of values in the layer that training can adjust:
     * one weight per input for each neuron, plus one bias per neuron.
     * This depends only on the layer's size, so build() does not need to have been called.
     */
    getParameterCount() {
        return this._numNeurons * this._inputSize + this._numNeurons;
    }
    /** Number of values coming into the layer. */
    get inputSize() {
        return this._inputSize;
    }
    /** Number of neurons in the layer. */
    get numNeurons() {
        return this._numNeurons;
    }
    /**
     * The weight matrix, indexed [neuron][input].
     * @throws Error if build() has not been called yet.
     */
    get weights() {
        if (this._weights === undefined) {
            throw new Error("HiddenLayer weights have not been created yet; call build() first");
        }
        return this._weights;
    }
    /**
     * The bias vector, one value per neuron.
     * @throws Error if build() has not been called yet.
     */
    get biases() {
        if (this._biases === undefined) {
            throw new Error("HiddenLayer biases have not been created yet; call build() first");
        }
        return this._biases;
    }
    /**
     * The inputs from the most recent calculate() call.
     * @throws Error if calculate() has not been called yet.
     */
    get inputs() {
        if (this._inputs === undefined) {
            throw new Error("HiddenLayer has no stored inputs yet; call calculate() first");
        }
        return this._inputs;
    }
    /**
     * Each neuron's output (after ReLU) from the most recent calculate() call.
     * @throws Error if calculate() has not been called yet.
     */
    get activations() {
        if (this._activations === undefined) {
            throw new Error("HiddenLayer has no stored activations yet; call calculate() first");
        }
        return this._activations;
    }
}
/**
 * Rectified Linear Unit: passes positive values through unchanged and
 * turns negative values into 0.
 */
function relu(x) {
    return Math.max(0, x);
}
//# sourceMappingURL=HiddenLayer.js.map