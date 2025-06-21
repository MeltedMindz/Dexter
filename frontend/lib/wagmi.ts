import { http, createConfig } from 'wagmi'
import { base, mainnet } from 'wagmi/chains'
import { coinbaseWallet, injected, walletConnect } from 'wagmi/connectors'

const projectId = process.env.NEXT_PUBLIC_WC_PROJECT_ID || '130fac92acf3681422903f821a58922a'
const alchemyApiKey = process.env.NEXT_PUBLIC_ALCHEMY_API_KEY

// Use public RPC endpoints as primary if no Alchemy key, or as fallbacks
const getBaseRpcUrl = () => {
  if (process.env.NEXT_PUBLIC_BASE_RPC_URL) {
    return process.env.NEXT_PUBLIC_BASE_RPC_URL
  }
  if (alchemyApiKey && alchemyApiKey !== 'demo' && alchemyApiKey !== 'demo-key-needs-replacement') {
    return `https://base-mainnet.g.alchemy.com/v2/${alchemyApiKey}`
  }
  return process.env.NEXT_PUBLIC_BASE_RPC_FALLBACK || 'https://mainnet.base.org'
}

const getMainnetRpcUrl = () => {
  if (process.env.NEXT_PUBLIC_MAINNET_RPC_URL) {
    return process.env.NEXT_PUBLIC_MAINNET_RPC_URL
  }
  if (alchemyApiKey && alchemyApiKey !== 'demo' && alchemyApiKey !== 'demo-key-needs-replacement') {
    return `https://eth-mainnet.g.alchemy.com/v2/${alchemyApiKey}`
  }
  return process.env.NEXT_PUBLIC_MAINNET_RPC_FALLBACK || 'https://eth.llamarpc.com'
}

const baseRpcUrl = getBaseRpcUrl()
const mainnetRpcUrl = getMainnetRpcUrl()

export const config = createConfig({
  chains: [base, mainnet],
  connectors: typeof window !== 'undefined' ? [
    injected(),
    coinbaseWallet({
      appName: 'Dexter Protocol',
    }),
    walletConnect({ 
      projectId,
      metadata: {
        name: 'Dexter Protocol',
        description: 'AI-Powered Liquidity Management',
        url: 'https://www.dexteragent.com',
        icons: ['https://www.dexteragent.com/favicon.ico']
      }
    }),
  ] : [],
  transports: {
    [base.id]: http(baseRpcUrl),
    [mainnet.id]: http(mainnetRpcUrl),
  },
  ssr: true,
})