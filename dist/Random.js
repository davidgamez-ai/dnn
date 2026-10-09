/**
 * Seedable random number generation. Math.random() cannot be seeded, so
 * runs that use it cannot be repeated. Everything in the project that needs
 * random numbers should use these functions instead.
 */
/** The generator's internal state, a 32-bit unsigned integer. Unseeded runs start from a random state. */
let state = Math.floor(Math.random() * 0x100000000) >>> 0;
/**
 * Restarts the generator from the given seed. After this, the sequence of
 * numbers returned by random() and randomNormal() is always the same for the same seed.
 *
 * @param seed An integer from 0 to 4294967295, usually Hyperparameters.randomSeed.
 */
export function setSeed(seed) {
    state = seed >>> 0;
}
/**
 * Returns a random number from 0 (inclusive) to 1 (exclusive), like
 * Math.random(), using the Mulberry32 algorithm.
 */
export function random() {
    state = (state + 0x6d2b79f5) >>> 0;
    let t = state;
    t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return ((t ^ (t >>> 14)) >>> 0) / 0x100000000;
}
/**
 * Returns a random number from the standard normal distribution
 * (mean 0, standard deviation 1), using the Box-Muller transform.
 */
export function randomNormal() {
    // 1 - random() is in (0, 1], which avoids taking the log of 0.
    const u1 = 1 - random();
    const u2 = random();
    return Math.sqrt(-2 * Math.log(u1)) * Math.cos(2 * Math.PI * u2);
}
//# sourceMappingURL=Random.js.map