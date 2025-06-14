import { http, createConfig } from 'wagmi'
import { base, mainnet } from 'wagmi/chains'
import { coinbaseWallet, injected, walletConnect } from 'wagmi/connectors'

const projectId = process.env.NEXT_PUBLIC_WC_PROJECT_ID || '130fac92acf3681422903f821a58922a'
const alchemyApiKey = process.env.NEXT_PUBLIC_ALCHEMY_API_KEY || 'ory0F2cLFNIXsovAmrtJj'
const baseRpcUrl = process.env.NEXT_PUBLIC_BASE_RPC_URL || `https://base-mainnet.g.alchemy.com/v2/${alchemyApiKey}`
const mainnetRpcUrl = process.env.NEXT_PUBLIC_MAINNET_RPC_URL || `https://eth-mainnet.g.alchemy.com/v2/${alchemyApiKey}`

export const config = createConfig({
  chains: [base, mainnet],
  connectors: [
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
  ],
  transports: {
    [base.id]: http(baseRpcUrl),
    [mainnet.id]: http(mainnetRpcUrl),
  },
  ssr: true,
})