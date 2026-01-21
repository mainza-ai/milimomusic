import React, { type ReactNode } from 'react';
import { motion } from 'framer-motion';
import { clsx, type ClassValue } from 'clsx';
import { twMerge } from 'tailwind-merge';

function cn(...inputs: ClassValue[]) {
    return twMerge(clsx(inputs));
}

interface GlassCardProps {
    children: ReactNode;
    className?: string;
    delay?: number;
}

export const GlassCard: React.FC<GlassCardProps> = ({ children, className, delay = 0 }) => {
    return (
        <motion.div
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.5, delay, ease: "easeOut" }}
            className={cn(
                "glass-panel rounded-2xl p-6 border border-white/20",
                "bg-white/40 shadow-xl backdrop-blur-xl",
                "hover:shadow-2xl transition-shadow duration-500",
                className
            )}
        >
            {children}
        </motion.div>
    );
};
