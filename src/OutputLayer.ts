import DEBUG from "./Debug.js";
import { randomNormal } from "./Random.js";

/**
 * The final, fully connected layer of the network. It has one neuron per
 * class (10 for the digits 0-9), and is intended to use the softmax
 * activation function so that its outputs can be read as probabilities.
 * The weight initialization in build() is chosen for this.
 */
export class OutputLayer {
    private readonly _inputSize: number;
    private readonly _numNeurons: number;
    private _weights: number[][] | undefined;
    private _biases: number[] | undefined;

    // From the most recent call to calculate(), kept for backpropagation.
    private _inputs: number[] | undefined;
    private _activations: number[] | undefined;

    // From the most recent call to calculateGradients().
    private _weightGradients: number[][] | undefined;
    private _biasGradients: number[] | undefined;

    /**
     * @param inputSize Number of values coming into the layer, i.e. the number of neurons in the last hidden layer.
     * @param numNeurons Number of neurons in the layer, one per class, e.g. 10 for the digits 0-9.
     */
    constructor(inputSize: number, numNeurons: number) {
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
    build(): void {
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
            const row: number[] = [];
            for (let i = 0; i < this._inputSize; i++) {
                row.push(randomNormal() * standardDeviation);
            }
            this._weights.push(row);
        }
        this._biases = new Array<number>(this._numNeurons).fill(0);
    }

    /**
     * Feeds a set of inputs through the layer and returns the probability of
     * each class. For each neuron, every input is multiplied by its weight,
     * the products are summed and the bias is added, giving the neuron's logit.
     * Softmax then turns the logits into probabilities.
     *
     * The inputs and the outputs (activations) are stored, replacing those from
     * the previous call. Backpropagation needs them to calculate this layer's
     * gradients.
     *
     * @param inputs Values coming into the layer; must have inputSize elements.
     * @returns One probability per neuron (class). The values are between 0 and 1 and add up to 1.
     * @throws RangeError if the number of inputs does not match inputSize.
     * @throws Error if build() has not been called yet.
     */
    calculate(inputs: readonly number[]): number[] {
        if (inputs.length !== this._inputSize) {
            throw new RangeError(`Expected ${this._inputSize} inputs, got ${inputs.length}`);
        }
        const weights = this.weights;
        const biases = this.biases;

        const logits: number[] = [];
        for (let n = 0; n < this._numNeurons; n++) {
            const neuronWeights = weights[n]!;
            let sum = 0;
            for (let i = 0; i < this._inputSize; i++) {
                sum += inputs[i]! * neuronWeights[i]!;
            }
            sum += biases[n]!;
            logits.push(sum);
        }

        if (DEBUG.OUTPUT_LOGITS) {
            console.log("Output layer logits:", logits);
        }
        const probabilities = softmax(logits);

        // Store copies, so changes the caller makes to either array do not alter them.
        this._inputs = [...inputs];
        this._activations = probabilities;
        return [...probabilities];
    }

    /**
     * Calculates the gradients for the most recent inputs passed to calculate()
     * and stores them in weightGradients and biasGradients, replacing those
     * from the previous call.
     *
     * Each neuron n works out
     *
     *     logit[n] = sum over i of ( weight[n][i] * input[i] ) + bias[n]
     *
     * The error signal for neuron n tells us how much the loss changes when
     * logit[n] changes. Using the chain rule:
     *
     *     weight gradient[n][i] = errorSignal[n] * input[i]
     *     bias gradient[n]      = errorSignal[n]
     *
     * So a weight changes the loss more when its neuron's error is large and
     * when the input it multiplies is large. If the input is 0, the weight had
     * no effect on this image, so its gradient is 0.
     *
     * @param errorSignals One error signal (probability - target) per neuron.
     * @throws RangeError if there is not one error signal per neuron.
     * @throws Error if calculate() has not been called yet.
     */
    calculateGradients(errorSignals: readonly number[]): void {
        if (errorSignals.length !== this._numNeurons) {
            throw new RangeError(`Expected ${this._numNeurons} error signals (one per neuron), got ${errorSignals.length}`);
        }
        const inputs = this.inputs;

        // Step 1: weight gradients, one per weight.
        const weightGradients: number[][] = [];
        for (let n = 0; n < this._numNeurons; n++) {
            const neuronWeightGradients: number[] = [];
            for (let i = 0; i < this._inputSize; i++) {
                neuronWeightGradients.push(errorSignals[n]! * inputs[i]!);
            }
            weightGradients.push(neuronWeightGradients);
        }

        // Step 2: bias gradients, one per neuron. A bias is added directly to
        // the logit (as if its input were always 1), so its gradient is just
        // the error signal.
        const biasGradients: number[] = [];
        for (let n = 0; n < this._numNeurons; n++) {
            biasGradients.push(errorSignals[n]!);
        }

        this._weightGradients = weightGradients;
        this._biasGradients = biasGradients;
    }

    /** Number of values coming into the layer. */
    get inputSize(): number {
        return this._inputSize;
    }

    /** Number of neurons in the layer. */
    get numNeurons(): number {
        return this._numNeurons;
    }

    /**
     * The weight matrix, indexed [neuron][input].
     * @throws Error if build() has not been called yet.
     */
    get weights(): readonly (readonly number[])[] {
        if (this._weights === undefined) {
            throw new Error("OutputLayer weights have not been created yet; call build() first");
        }
        return this._weights;
    }

    /**
     * The bias vector, one value per neuron.
     * @throws Error if build() has not been called yet.
     */
    get biases(): readonly number[] {
        if (this._biases === undefined) {
            throw new Error("OutputLayer biases have not been created yet; call build() first");
        }
        return this._biases;
    }

    /**
     * The inputs from the most recent calculate() call.
     * @throws Error if calculate() has not been called yet.
     */
    get inputs(): readonly number[] {
        if (this._inputs === undefined) {
            throw new Error("OutputLayer has no stored inputs yet; call calculate() first");
        }
        return this._inputs;
    }

    /**
     * Each neuron's output (the probabilities, after softmax) from the most
     * recent calculate() call.
     * @throws Error if calculate() has not been called yet.
     */
    get activations(): readonly number[] {
        if (this._activations === undefined) {
            throw new Error("OutputLayer has no stored activations yet; call calculate() first");
        }
        return this._activations;
    }

    /**
     * The weight gradients from the most recent calculateGradients() call,
     * indexed [neuron][input] like the weight matrix.
     * @throws Error if calculateGradients() has not been called yet.
     */
    get weightGradients(): readonly (readonly number[])[] {
        if (this._weightGradients === undefined) {
            throw new Error("OutputLayer has no weight gradients yet; call calculateGradients() first");
        }
        return this._weightGradients;
    }

    /**
     * The bias gradients from the most recent calculateGradients() call, one per neuron.
     * @throws Error if calculateGradients() has not been called yet.
     */
    get biasGradients(): readonly number[] {
        if (this._biasGradients === undefined) {
            throw new Error("OutputLayer has no bias gradients yet; call calculateGradients() first");
        }
        return this._biasGradients;
    }
}

/**
 * Converts logits into probabilities: each output is e^logit divided by the
 * sum of e^logit over all logits, so the outputs are positive and add up to 1.
 */
function softmax(logits: readonly number[]): number[] {
    // Subtracting the largest logit gives the same result but stops Math.exp
    // overflowing to Infinity when logits are large.
    const max = Math.max(...logits);
    const exps = logits.map(logit => Math.exp(logit - max));
    const total = exps.reduce((sum, value) => sum + value, 0);
    return exps.map(value => value / total);
}
