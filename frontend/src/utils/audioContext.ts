export const getAudioContext = (): AudioContext => {
    const w = window as any;
    if (!w._milimoAudioContext) {
        const AudioContextClass = w.AudioContext || w.webkitAudioContext;
        w._milimoAudioContext = new AudioContextClass();
    }
    return w._milimoAudioContext;
};
