import React from 'react';

interface MilimoLogoProps {
    size?: 'sm' | 'md' | 'lg';
    showText?: boolean;
    className?: string;
}

export const MilimoLogo: React.FC<MilimoLogoProps> = ({
    size = 'md',
    showText = true,
    className = ''
}) => {
    const sizeMap = {
        sm: { img: 'w-7 h-7', title: 'text-xs', sub: 'text-[9px]' },
        md: { img: 'w-9 h-9', title: 'text-sm', sub: 'text-[10px]' },
        lg: { img: 'w-12 h-12', title: 'text-lg', sub: 'text-xs' }
    };

    const currentSize = sizeMap[size];

    return (
        <div className={`flex items-center space-x-3 select-none ${className}`}>
            {/* Logo Image with Apple App Icon container */}
            <div className={`relative ${currentSize.img} rounded-xl overflow-hidden shadow-apple-sm flex-shrink-0 border border-black/10 dark:border-white/15 bg-gradient-to-tr from-teal-500/10 to-cyan-500/10 p-0.5 group`}>
                <img
                    src="/milimo_logo.png"
                    alt="Milimo Music Logo"
                    className="w-full h-full object-cover rounded-[10px] transform group-hover:scale-105 transition-transform duration-300"
                    onError={(e) => {
                        (e.target as HTMLElement).style.display = 'none';
                    }}
                />
            </div>

            {showText && (
                <div className="flex flex-col">
                    <div className="flex items-center space-x-1.5">
                        <span className={`${currentSize.title} font-extrabold tracking-tight text-slate-900 dark:text-white font-sans`}>
                            Milimo Music
                        </span>
                        <span className="text-[9px] font-mono px-1.5 py-0.2 rounded-full bg-teal-500/10 dark:bg-teal-500/20 text-teal-700 dark:text-teal-300 font-semibold border border-teal-500/20">
                            v2
                        </span>
                    </div>
                    <span className={`${currentSize.sub} font-medium text-slate-500 dark:text-slate-400 tracking-tight`}>
                        AI Music Production DAW
                    </span>
                </div>
            )}
        </div>
    );
};
