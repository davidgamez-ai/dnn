import { Hyperparameters } from "./Hyperparameters.js";

/** The running averages AdamW keeps for one array of parameters. */
interface Moments {
    /** First moment: running average of the gradients. */
    readonly m: number[];
    /** Second moment: running average of the squared gradients. */
    readonly v: number[];
}

/**
 * The AdamW optimizer (Loshchilov & Hutter, "Decoupled Weight Decay
 * Regularization", 2019). It decides how to change each parameter (weight or
 * bias) given its gradient.
 *
 * Plain gradient descent moves every parameter by learningRate * gradient.
 * AdamW improves on this in three ways:
 *
 *   1. Momentum: it follows a running average of the gradients (m), which
 *      smooths out the noise from one batch to the next.
 *   2. Per-parameter step sizes: it divides by the square root of a running
 *      average of the squared gradients (v). Parameters with consistently
 *      large gradients take smaller steps, and those with small gradients take
 *      larger ones, so each step is roughly learningRate in size.
 *   3. Decoupled weight decay: it shrinks each weight slightly towards 0 on
 *      every step, separately from the gradient. (In the original Adam,
 *      weight decay was mixed into the gradient, which made it much weaker for
 *      parameters with large gradients.)
 *
 * For each parameter p with gradient g, on step t:
 *
 *     m = beta1 * m + (1 - beta1) * g            running average of gradients
 *     v = beta2 * v + (1 - beta2) * g * g        running average of squared gradients
 *     mHat = m / (1 - beta1^t)                   bias correction (see below)
 *     vHat = v / (1 - beta2^t)
 *     p = p - learningRate * weightDecay * p     weight decay
 *     p = p - learningRate * mHat / (sqrt(vHat) + epsilon)
 *
 * Bias correction: m and v start at 0, so for the first few steps they are
 * too small. Dividing by (1 - beta^t) scales them back up; as t grows the
 * divisor approaches 1 and the correction fades away.
 *
 * Usage, once per batch:
 *
 *     optimizer.step();                                    // start a new step (t = t + 1)
 *     optimizer.updateMatrix(weights, weightGradients);    // for each layer
 *     optimizer.updateVector(biases, biasGradients);
 *
 * AdamW remembers m and v separately for every array it is given, so always
 * pass the same weight and bias arrays (updated in place) from step to step.
 *
 * The settings (learningRate, beta1, beta2, epsilon and weightDecay) come
 * from the hyperparameters, and are read each time parameters are updated.
 */
export class AdamW {
    /** Number of steps taken so far (t in the formulas). */
    private _stepCount = 0;

    /**
     * m and v for each parameter array, looked up by the array itself. A
     * WeakMap lets an array's moments be discarded automatically once the array
     * is no longer used, e.g. after a layer is rebuilt.
     */
    private readonly _moments = new WeakMap<number[], Moments>();

    /**
     * @param hyperparameters Supplies learningRate, beta1, beta2, epsilon and weightDecay.
     */
    constructor(readonly hyperparameters: Hyperparameters) {}

    /**
     * Starts a new optimization step. Call it once per batch, before updating
     * any parameters for that batch.
     */
    step(): void {
        this._stepCount++;
    }

    /**
     * Updates a weight matrix in place, one row at a time.
     *
     * @param weights The weights to update, indexed [neuron][input].
     * @param gradients The gradients for these weights, the same shape as weights.
     * @param applyWeightDecay Whether to shrink these parameters towards 0.
     * Normally true for weights.
     * @throws RangeError if the shapes do not match.
     * @throws Error if step() has not been called yet.
     */
    updateMatrix(weights: number[][], gradients: readonly (readonly number[])[], applyWeightDecay = true): void {
        if (gradients.length !== weights.length) {
            throw new RangeError(`Expected ${weights.length} rows of gradients, got ${gradients.length}`);
        }
        for (let n = 0; n < weights.length; n++) {
            this.updateVector(weights[n]!, gradients[n]!, applyWeightDecay);
        }
    }

    /**
     * Updates an array of parameters in place, e.g. a layer's biases or one
     * row of its weights.
     *
     * @param parameters The parameters to update.
     * @param gradients The gradient for each parameter.
     * @param applyWeightDecay Whether to shrink these parameters towards 0.
     * Normally false for biases, which do not make the network overfit the
     * way large weights can.
     * @throws RangeError if the lengths do not match.
     * @throws Error if step() has not been called yet.
     */
    updateVector(parameters: number[], gradients: readonly number[], applyWeightDecay = false): void {
        if (this._stepCount === 0) {
            throw new Error("Call step() before updating parameters");
        }
        if (gradients.length !== parameters.length) {
            throw new RangeError(`Expected ${parameters.length} gradients, got ${gradients.length}`);
        }

        // Find this array's running averages, creating them (all 0) the first time.
        let moments = this._moments.get(parameters);
        if (moments === undefined) {
            moments = {
                m: new Array<number>(parameters.length).fill(0),
                v: new Array<number>(parameters.length).fill(0),
            };
            this._moments.set(parameters, moments);
        }
        const { m, v } = moments;

        const { learningRate, beta1, beta2, epsilon, weightDecay } = this.hyperparameters;

        // Bias corrections for this step; the same for every parameter.
        const t = this._stepCount;
        const mCorrection = 1 - Math.pow(beta1, t);
        const vCorrection = 1 - Math.pow(beta2, t);

        for (let i = 0; i < parameters.length; i++) {
            const g = gradients[i]!;

            // Update the running averages.
            m[i] = beta1 * m[i]! + (1 - beta1) * g;
            v[i] = beta2 * v[i]! + (1 - beta2) * g * g;

            // Correct them for starting at 0.
            const mHat = m[i]! / mCorrection;
            const vHat = v[i]! / vCorrection;

            // Decoupled weight decay: shrink the parameter towards 0.
            if (applyWeightDecay) {
                parameters[i]! -= learningRate * weightDecay * parameters[i]!;
            }

            // Adam step: move against the smoothed gradient, scaled per parameter.
            parameters[i]! -= learningRate * mHat / (Math.sqrt(vHat) + epsilon);
        }
    }

    /** Number of steps taken so far. */
    get stepCount(): number {
        return this._stepCount;
    }
}
