import { http, createConfig } from 'wagmi'
import { base, mainnet } from 'wagmi/chains'
import { coinbaseWallet, injected, walletConnect } from 'wagmi/connectors'

const projectId = process.env.NEXT_PUBLIC_WC_PROJECT_ID || 'demo-project-id'
const baseRpcUrl = process.env.NEXT_PUBLIC_BASE_RPC_URL || 'https://mainnet.base.org'

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
        url: 'https://dexteragent.com',
        icons: ['https://dexteragent.com/favicon.ico']
      }
    }),
  ],
  transports: {
    [base.id]: http(baseRpcUrl),
    [mainnet.id]: http(),
  },
  ssr: true,
})