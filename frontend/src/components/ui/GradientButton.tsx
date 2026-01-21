import React from 'react';
import { motion } from 'framer-motion';
import { Loader2 } from 'lucide-react';
import { clsx, type ClassValue } from 'clsx';
import { twMerge } from 'tailwind-merge';

function cn(...inputs: ClassValue[]) {
    return twMerge(clsx(inputs));
}

interface GradientButtonProps extends React.ButtonHTMLAttributes<HTMLButtonElement> {
    children: React.ReactNode;
    isLoading?: boolean;
    variant?: 'primary' | 'glass';
}

export const GradientButton: React.FC<GradientButtonProps> = ({
    children,
    isLoading,
    className,
    variant = 'primary',
    disabled,
    ...props
}) => {
    const baseStyles = "relative px-6 py-3 rounded-xl font-semibold transition-all duration-300 transform active:scale-95 flex items-center justify-center gap-2";

    const variants = {
        primary: "text-white shadow-lg shadow-cyan-500/30 hover:shadow-cyan-500/50 bg-gradient-to-r from-cyan-600 via-teal-500 to-cyan-600 bg-[length:200%_auto] hover:bg-right",
        glass: "text-slate-800 bg-white/50 hover:bg-white/70 border border-white/40 shadow-sm hover:shadow-md"
    };

    return (
        <motion.button
            whileHover={{ y: -2 }}
            className={cn(baseStyles, variants[variant], className, (disabled || isLoading) && "opacity-70 cursor-not-allowed")}
            disabled={disabled || isLoading}
            onClick={props.onClick as any}
            type={props.type ? (props.type as "button" | "submit" | "reset") : "button"}
        >
            {isLoading ? (
                <>
                    <Loader2 className="w-5 h-5 animate-spin" />
                    <span>Processing...</span>
                </>
            ) : children}
        </motion.button>
    );
};
