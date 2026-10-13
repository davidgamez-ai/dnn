/**
 * Settings that control how the neural network is built and trained.
 * Unlike the network's weights and biases, these are chosen before
 * training starts and are not learned from the data.
 */
export class Hyperparameters {
    // Set in the constructor through the setters below, which TypeScript
    // cannot see, hence the ! (definite assignment) markers.
    _learningRate;
    _epochs;
    _batchSize;
    _hiddenLayerSizes;
    _dataSplit;
    _randomSeed;
    _beta1;
    _beta2;
    _epsilon;
    _weightDecay;
    /**
     * Creates a set of hyperparameters. Any value that is not supplied
     * takes its default.
     */
    constructor({ learningRate = 0.001, epochs = 50, batchSize = 32, hiddenLayerSizes = [32], dataSplit = { train: 0.7, validation: 0.15, test: 0.15 }, randomSeed = 42, beta1 = 0.9, beta2 = 0.999, epsilon = 1e-8, weightDecay = 0.01, } = {}) {
        // Assigning through the setters means the constructor and the setters share the same checks.
        this.learningRate = learningRate;
        this.epochs = epochs;
        this.batchSize = batchSize;
        this.hiddenLayerSizes = hiddenLayerSizes;
        this.dataSplit = dataSplit;
        this.randomSeed = randomSeed;
        this.beta1 = beta1;
        this.beta2 = beta2;
        this.epsilon = epsilon;
        this.weightDecay = weightDecay;
    }
    /**
     * Step size used when updating the weights after each batch. The AdamW
     * optimizer usually works well around 0.001.
     */
    get learningRate() {
        return this._learningRate;
    }
    /** @throws RangeError if the value is not greater than 0. */
    set learningRate(value) {
        if (!(value > 0)) {
            throw new RangeError(`learningRate must be greater than 0, got ${value}`);
        }
        this._learningRate = value;
    }
    /** Number of complete passes through the training data. */
    get epochs() {
        return this._epochs;
    }
    /** @throws RangeError if the value is not a positive integer. */
    set epochs(value) {
        if (!Number.isInteger(value) || value < 1) {
            throw new RangeError(`epochs must be a positive integer, got ${value}`);
        }
        this._epochs = value;
    }
    /** Number of training examples processed before the weights are updated. */
    get batchSize() {
        return this._batchSize;
    }
    /** @throws RangeError if the value is not a positive integer. */
    set batchSize(value) {
        if (!Number.isInteger(value) || value < 1) {
            throw new RangeError(`batchSize must be a positive integer, got ${value}`);
        }
        this._batchSize = value;
    }
    /**
     * Number of neurons in each hidden layer, from input side to output side.
     * Returns a copy, so changing it does not affect these hyperparameters.
     */
    get hiddenLayerSizes() {
        return [...this._hiddenLayerSizes];
    }
    /**
     * Stores a copy, so changing the original array afterwards has no effect.
     * @throws RangeError if any size is not a positive integer.
     */
    set hiddenLayerSizes(value) {
        if (value.some(size => !Number.isInteger(size) || size < 1)) {
            throw new RangeError(`hiddenLayerSizes must all be positive integers, got [${value.join(", ")}]`);
        }
        this._hiddenLayerSizes = [...value];
    }
    /**
     * How the dataset is divided into training, validation and test sets.
     * Returns a copy, so changing it does not affect these hyperparameters.
     */
    get dataSplit() {
        return { ...this._dataSplit };
    }
    /**
     * Stores a copy, so changing the original object afterwards has no effect.
     * @throws RangeError if train is not greater than 0, validation or test is
     * negative, or the three do not add up to 1.
     */
    set dataSplit(value) {
        const { train, validation, test } = value;
        if (!(train > 0) || !(validation >= 0) || !(test >= 0)) {
            throw new RangeError(`dataSplit.train must be greater than 0 and validation and test must not be negative, got ${train}/${validation}/${test}`);
        }
        // Allow for floating-point rounding, e.g. 0.7 + 0.15 + 0.15 is not exactly 1.
        if (Math.abs(train + validation + test - 1) > 1e-9) {
            throw new RangeError(`dataSplit fractions must add up to 1, got ${train}/${validation}/${test}`);
        }
        this._dataSplit = { train, validation, test };
    }
    /**
     * Starting value for the random number generator. Using the same seed
     * gives the same data split and the same starting weights, so a run can
     * be repeated exactly. Change it to get a different random run.
     */
    get randomSeed() {
        return this._randomSeed;
    }
    /** @throws RangeError if the value is not an integer from 0 to 4294967295. */
    set randomSeed(value) {
        if (!Number.isInteger(value) || value < 0 || value > 0xffffffff) {
            throw new RangeError(`randomSeed must be an integer from 0 to 4294967295, got ${value}`);
        }
        this._randomSeed = value;
    }
    /**
     * AdamW: how much of the previous average gradient to keep each step
     * (the "momentum"). Higher values smooth the gradients more.
     */
    get beta1() {
        return this._beta1;
    }
    /** @throws RangeError if the value is not at least 0 and less than 1. */
    set beta1(value) {
        if (!(value >= 0 && value < 1)) {
            throw new RangeError(`beta1 must be at least 0 and less than 1, got ${value}`);
        }
        this._beta1 = value;
    }
    /** AdamW: how much of the previous average squared gradient to keep each step. */
    get beta2() {
        return this._beta2;
    }
    /** @throws RangeError if the value is not at least 0 and less than 1. */
    set beta2(value) {
        if (!(value >= 0 && value < 1)) {
            throw new RangeError(`beta2 must be at least 0 and less than 1, got ${value}`);
        }
        this._beta2 = value;
    }
    /** AdamW: small number added to the denominator so it never divides by 0. */
    get epsilon() {
        return this._epsilon;
    }
    /** @throws RangeError if the value is not greater than 0. */
    set epsilon(value) {
        if (!(value > 0)) {
            throw new RangeError(`epsilon must be greater than 0, got ${value}`);
        }
        this._epsilon = value;
    }
    /**
     * AdamW: how strongly weights are pulled towards 0 each step, to
     * discourage very large weights. 0 turns weight decay off.
     */
    get weightDecay() {
        return this._weightDecay;
    }
    /** @throws RangeError if the value is negative. */
    set weightDecay(value) {
        if (!(value >= 0)) {
            throw new RangeError(`weightDecay must not be negative, got ${value}`);
        }
        this._weightDecay = value;
    }
}
//# sourceMappingURL=Hyperparameters.js.map