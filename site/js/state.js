/**
 * Shared application state with a tiny event bus.
 *
 * The state holds the playhead, the playback controls and the story position. Any module reads it
 * with `state.get()`, changes it with `state.set(partial)` and listens with `state.on(key, fn)`.
 * There are no dependencies.
 *
 * Keys:
 *  - `frame`      {number}  current video frame, an integer
 *  - `playing`    {boolean} true while the replay runs
 *  - `speed`      {number}  playback speed multiplier, for example 0.25 to 4
 *  - `half`       {1|2}     half that the playhead is in
 *  - `mode`       {'public'|'local'} which data bundle is loaded
 *  - `activeStep` {{section: string, step: number, progress: number}} current scroll step, progress 0..1
 *  - `arena`      {boolean} true in arena mode
 *
 * @module state
 */

/**
 * @typedef {Object} ActiveStep
 * @property {string} section  Section id, for example 'engine'.
 * @property {number} step     Step index in the section, starting at 0.
 * @property {number} progress Scroll progress through the step, 0 to 1.
 */

/**
 * @typedef {Object} AppState
 * @property {number} frame
 * @property {boolean} playing
 * @property {number} speed
 * @property {1|2} half
 * @property {'public'|'local'} mode
 * @property {ActiveStep} activeStep
 * @property {boolean} arena
 */

/** @returns {AppState} A fresh default state. */
function defaults() {
  return {
    frame: 11185,
    playing: false,
    speed: 1,
    half: 1,
    mode: 'public',
    activeStep: { section: 'hero', step: 0, progress: 0 },
    arena: false,
  };
}

/**
 * Create a state store. The module exports one shared instance, `state`.
 * @returns {{get: function(): AppState, set: function(Partial<AppState>): void,
 *            on: function(string, function(*, AppState): void): function(): void}}
 */
export function createState() {
  let current = defaults();
  /** @type {Map<string, Set<Function>>} */
  const listeners = new Map();

  /**
   * Call every listener of one key. A throwing listener never stops the others.
   * @param {string} key
   * @param {*} value
   */
  function emit(key, value) {
    const set = listeners.get(key);
    if (!set) return;
    for (const fn of Array.from(set)) {
      try {
        fn(value, current);
      } catch (err) {
        console.error('state listener failed for "' + key + '"', err);
      }
    }
  }

  return {
    /**
     * Read the current state.
     * @returns {AppState} A shallow copy. Changing it does not change the state.
     */
    get() {
      return Object.assign({}, current);
    },

    /**
     * Merge a partial state. Emit one event for each changed key, then one `change` event.
     * Listeners get `(value, fullState)` for a key event and `(changedKeys, fullState)` for `change`.
     * A key counts as changed when `Object.is` says the new value differs. `activeStep` always
     * counts as changed when given, because its object is replaced.
     * @param {Partial<AppState>} partial
     */
    set(partial) {
      const next = Object.assign({}, current);
      const changed = [];
      for (const key of Object.keys(partial || {})) {
        const value = partial[key];
        const same = key === 'activeStep'
          ? false
          : Object.is(current[key], value);
        if (!same) {
          next[key] = value;
          changed.push(key);
        }
      }
      if (changed.length === 0) return;
      current = next;
      for (const key of changed) emit(key, current[key]);
      emit('change', changed);
    },

    /**
     * Subscribe to a key, or to `change` for any change.
     * @param {string} key  A state key or 'change'.
     * @param {function(*, AppState): void} fn
     * @returns {function(): void} Call it to unsubscribe.
     */
    on(key, fn) {
      if (!listeners.has(key)) listeners.set(key, new Set());
      listeners.get(key).add(fn);
      return function unsubscribe() {
        const set = listeners.get(key);
        if (set) set.delete(fn);
      };
    },
  };
}

/** The shared state store. */
export const state = createState();

export default state;
