'use client'

import { useState } from 'react'
import Image from 'next/image'

interface TokenIconProps {
  symbol: string
  name: string
  size?: 'sm' | 'md' | 'lg'
  className?: string
}

export function TokenIcon({ symbol, name, size = 'md', className = '' }: TokenIconProps) {
  const [imageError, setImageError] = useState(false)
  
  const sizeClasses = {
    sm: 'w-6 h-6 text-xs',
    md: 'w-8 h-8 text-sm',
    lg: 'w-12 h-12 text-lg'
  }

  // Try to load token image, fallback to symbol letter
  const tokenImageSrc = `/tokens/${symbol.toLowerCase()}.png`

  if (imageError) {
    return (
      <div 
        className={`${sizeClasses[size]} bg-primary border-2 border-black dark:border-white flex items-center justify-center ${className}`}
        title={name}
      >
        <span className="text-black font-bold font-mono">
          {symbol.charAt(0)}
        </span>
      </div>
    )
  }

  return (
    <div className={`${sizeClasses[size]} relative ${className}`} title={name}>
      <Image
        src={tokenImageSrc}
        alt={`${name} (${symbol})`}
        width={32}
        height={32}
        className="w-full h-full rounded-none border-2 border-black dark:border-white"
        onError={() => setImageError(true)}
        onLoad={() => setImageError(false)}
      />
    </div>
  )
}