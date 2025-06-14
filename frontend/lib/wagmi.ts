import { http, createConfig } from 'wagmi'
import { base, mainnet } from 'wagmi/chains'
import { coinbaseWallet, injected, walletConnect } from 'wagmi/connectors'

const projectId = process.env.NEXT_PUBLIC_WC_PROJECT_ID || '130fac92acf3681422903f821a58922a'
const infuraApiKey = process.env.NEXT_PUBLIC_INFURA_API_KEY || '3e212a0d9ac24bbd9570069de26846ae'
const baseRpcUrl = process.env.NEXT_PUBLIC_BASE_RPC_URL || `https://base-mainnet.infura.io/v3/${infuraApiKey}`
const mainnetRpcUrl = `https://mainnet.infura.io/v3/${infuraApiKey}`

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