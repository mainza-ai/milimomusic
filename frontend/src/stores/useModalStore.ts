import { create } from 'zustand';
import type { Job } from '../api';

interface ModalStoreState {
    // Cover & Remix Studio (MuLaCover)
    isCoverStudioOpen: boolean;
    coverStudioTrack: Job | null;
    coverStudioMode: 'audio' | 'midi';
    coverStudioStemPath?: string;
    openCoverStudio: (track?: Job | null, mode?: 'audio' | 'midi', stemPath?: string) => void;
    closeCoverStudio: () => void;

    // Engine Quick-Switcher (Ctrl+E)
    isEngineSwitcherOpen: boolean;
    openEngineSwitcher: () => void;
    closeEngineSwitcher: () => void;

    // Voice Conversion & Timbre Transfer
    isVoiceConvertOpen: boolean;
    voiceConvertTrack: Job | null;
    voiceConvertStemPath?: string;
    openVoiceConvert: (track?: Job | null, stemPath?: string) => void;
    closeVoiceConvert: () => void;

    // Track Extension Studio
    isExtendTrackOpen: boolean;
    extendTrackJob: Job | null;
    openExtendTrack: (track: Job) => void;
    closeExtendTrack: () => void;
}

export const useModalStore = create<ModalStoreState>((set) => ({
    // Cover Studio Modal
    isCoverStudioOpen: false,
    coverStudioTrack: null,
    coverStudioMode: 'audio',
    coverStudioStemPath: undefined,
    openCoverStudio: (track = null, mode = 'audio', stemPath) => set({
        isCoverStudioOpen: true,
        coverStudioTrack: track,
        coverStudioMode: mode,
        coverStudioStemPath: stemPath,
    }),
    closeCoverStudio: () => set({
        isCoverStudioOpen: false,
        coverStudioTrack: null,
        coverStudioStemPath: undefined,
    }),

    // Engine Quick Switcher
    isEngineSwitcherOpen: false,
    openEngineSwitcher: () => set({ isEngineSwitcherOpen: true }),
    closeEngineSwitcher: () => set({ isEngineSwitcherOpen: false }),

    // Voice Convert Modal
    isVoiceConvertOpen: false,
    voiceConvertTrack: null,
    voiceConvertStemPath: undefined,
    openVoiceConvert: (track = null, stemPath) => set({
        isVoiceConvertOpen: true,
        voiceConvertTrack: track,
        voiceConvertStemPath: stemPath,
    }),
    closeVoiceConvert: () => set({
        isVoiceConvertOpen: false,
        voiceConvertTrack: null,
        voiceConvertStemPath: undefined,
    }),

    // Track Extension Modal
    isExtendTrackOpen: false,
    extendTrackJob: null,
    openExtendTrack: (track: Job) => set({
        isExtendTrackOpen: true,
        extendTrackJob: track,
    }),
    closeExtendTrack: () => set({
        isExtendTrackOpen: false,
        extendTrackJob: null,
    }),
}));

