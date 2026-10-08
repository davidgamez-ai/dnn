import DEBUG from "./Debug.js";
import { randomNormal } from "./Random.js";
/**
 * The final, fully connected layer of the network. It has one neuron per
 * class (10 for the digits 0-9), and is intended to use the softmax
 * activation function so that its outputs can be read as probabilities.
 * The weight initialization in build() is chosen for this.
 */
export class OutputLayer {
    _inputSize;
    _numNeurons;
    _weights;
    _biases;
    /**
     * @param inputSize Number of values coming into the layer, i.e. the number of neurons in the last hidden layer.
     * @param numNeurons Number of neurons in the layer, one per class, e.g. 10 for the digits 0-9.
     */
    constructor(inputSize, numNeurons) {
        this._inputSize = inputSize;
        this._numNeurons = numNeurons;
    }
    /**
     * Checks the layer's sizes, then creates the weight matrix and bias vector
     * and sets their starting values. Calling it again discards the current
     * weights and biases and starts afresh.
     *
     * The matrix has one row per neuron and one column per input, so
     * weights[n][i] is the weight neuron n gives to input i.
     *
     * Uses Xavier (Glorot) initialization: values are drawn from a normal
     * distribution with mean 0 and standard deviation
     * sqrt(2 / (inputSize + numNeurons)). This suits softmax outputs better
     * than the He initialization used for the ReLU hidden layers, which would
     * start the outputs further from an even spread across the classes.
     *
     * There is one bias per neuron, so biases[n] is added to neuron n's
     * weighted sum. Biases start at 0, so no class is favoured at the start.
     *
     * @throws RangeError if inputSize is not a positive integer, or numNeurons
     * is not an integer of at least 2.
     */
    build() {
        if (!Number.isInteger(this._inputSize) || this._inputSize < 1) {
            throw new RangeError(`inputSize must be a positive integer, got ${this._inputSize}`);
        }
        // Softmax over a single neuron always outputs 1, so it could not tell classes apart.
        if (!Number.isInteger(this._numNeurons) || this._numNeurons < 2) {
            throw new RangeError(`numNeurons must be an integer of at least 2 (one per class), got ${this._numNeurons}`);
        }
        const standardDeviation = Math.sqrt(2 / (this._inputSize + this._numNeurons));
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
     * Feeds a set of inputs through the layer and returns the probability of
     * each class. For each neuron, every input is multiplied by its weight,
     * the products are summed and the bias is added, giving the neuron's logit.
     * Softmax then turns the logits into probabilities.
     *
     * @param inputs Values coming into the layer; must have inputSize elements.
     * @returns One probability per neuron (class). The values are between 0 and 1 and add up to 1.
     * @throws RangeError if the number of inputs does not match inputSize.
     * @throws Error if build() has not been called yet.
     */
    calculate(inputs) {
        if (inputs.length !== this._inputSize) {
            throw new RangeError(`Expected ${this._inputSize} inputs, got ${inputs.length}`);
        }
        const weights = this.weights;
        const biases = this.biases;
        const logits = [];
        for (let n = 0; n < this._numNeurons; n++) {
            const neuronWeights = weights[n];
            let sum = 0;
            for (let i = 0; i < this._inputSize; i++) {
                sum += inputs[i] * neuronWeights[i];
            }
            sum += biases[n];
            logits.push(sum);
        }
        if (DEBUG.OUTPUT_LOGITS) {
            console.log("Output layer logits:", logits);
        }
        return softmax(logits);
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
            throw new Error("OutputLayer weights have not been created yet; call build() first");
        }
        return this._weights;
    }
    /**
     * The bias vector, one value per neuron.
     * @throws Error if build() has not been called yet.
     */
    get biases() {
        if (this._biases === undefined) {
            throw new Error("OutputLayer biases have not been created yet; call build() first");
        }
        return this._biases;
    }
}
/**
 * Converts logits into probabilities: each output is e^logit divided by the
 * sum of e^logit over all logits, so the outputs are positive and add up to 1.
 */
function softmax(logits) {
    // Subtracting the largest logit gives the same result but stops Math.exp
    // overflowing to Infinity when logits are large.
    const max = Math.max(...logits);
    const exps = logits.map(logit => Math.exp(logit - max));
    const total = exps.reduce((sum, value) => sum + value, 0);
    return exps.map(value => value / total);
}
//# sourceMappingURL=OutputLayer.js.map