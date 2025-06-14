import { http, createConfig } from 'wagmi'
import { base, mainnet } from 'wagmi/chains'
import { coinbaseWallet, injected, walletConnect } from 'wagmi/connectors'

const projectId = process.env.NEXT_PUBLIC_WC_PROJECT_ID || 'demo-project-id'

export const config = createConfig({
  chains: [base, mainnet],
  connectors: [
    injected(),
    coinbaseWallet({
      appName: 'Dexter Protocol',
    }),
    walletConnect({ 
      projectId,
    }),
  ],
  transports: {
    [base.id]: http(),
    [mainnet.id]: http(),
  },
  ssr: true,
})