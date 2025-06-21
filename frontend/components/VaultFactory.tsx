'use client'

import React, { useState, useEffect } from 'react'
import { useAccount, useWriteContract, useWaitForTransactionReceipt } from 'wagmi'
import { parseEther, formatEther } from 'viem'
import { 
  Factory, 
  Zap, 
  Settings, 
  TrendingUp,
  Shield,
  Users,
  Info,
  ChevronRight,
  CheckCircle,
  AlertTriangle,
  Sparkles
} from 'lucide-react'

// Types
interface VaultTemplate {
  id: string
  name: string
  description: string
  features: string[]
  minDeposit: string
  managementFee: number
  performanceFee: number
  riskLevel: 'Low' | 'Medium' | 'High'
  aiEnabled: boolean
  icon: React.ReactNode
}

interface TokenInfo {
  address: string
  symbol: string
  name: string
  decimals: number
}

interface VaultConfig {
  strategyMode: 'MANUAL' | 'AI_ASSISTED' | 'FULLY_AUTOMATED'
  positionType: 'SINGLE_RANGE' | 'DUAL_POSITION' | 'MULTI_RANGE' | 'AI_OPTIMIZED'
  autoCompound: boolean
  rebalanceThreshold: number
  maxSlippage: number
}

const VAULT_TEMPLATES: VaultTemplate[] = [
  {
    id: 'basic',
    name: 'Basic Vault',
    description: 'Standard automated liquidity management with proven strategies',
    features: [
      'Single-range positions',
      'Auto-compounding',
      'Manual rebalancing',
      'Low gas optimization'
    ],
    minDeposit: '100',
    managementFee: 0.5,
    performanceFee: 10,
    riskLevel: 'Low',
    aiEnabled: false,
    icon: <Settings className="w-6 h-6" />
  },
  {
    id: 'gamma',
    name: 'Gamma-Style Vault', 
    description: 'Dual-position strategy inspired by Gamma Strategies',
    features: [
      'Base + Limit positions',
      'TWAP protection',
      'Proven dual-range strategy',
      'Balanced risk/reward'
    ],
    minDeposit: '500',
    managementFee: 0.75,
    performanceFee: 12.5,
    riskLevel: 'Medium',
    aiEnabled: false,
    icon: <TrendingUp className="w-6 h-6" />
  },
  {
    id: 'ai_optimized',
    name: 'AI Optimized Vault',
    description: 'Advanced AI-powered optimization with machine learning',
    features: [
      'ML-driven strategies',
      'Dynamic rebalancing',
      'Risk assessment',
      'Performance prediction'
    ],
    minDeposit: '1000',
    managementFee: 0.75,
    performanceFee: 15,
    riskLevel: 'Medium',
    aiEnabled: true,
    icon: <Zap className="w-6 h-6" />
  },
  {
    id: 'institutional',
    name: 'Institutional Vault',
    description: 'Enterprise-grade vault with advanced features and lower fees',
    features: [
      'Multi-range positions',
      'Custom fee structure',
      'Advanced analytics',
      'Dedicated support'
    ],
    minDeposit: '100000',
    managementFee: 0.25,
    performanceFee: 7.5,
    riskLevel: 'Low',
    aiEnabled: true,
    icon: <Shield className="w-6 h-6" />
  }
]

const POPULAR_TOKENS: TokenInfo[] = [
  { address: '0x...', symbol: 'ETH', name: 'Ethereum', decimals: 18 },
  { address: '0x...', symbol: 'USDC', name: 'USD Coin', decimals: 6 },
  { address: '0x...', symbol: 'USDT', name: 'Tether USD', decimals: 6 },
  { address: '0x...', symbol: 'WBTC', name: 'Wrapped Bitcoin', decimals: 8 },
  { address: '0x...', symbol: 'DAI', name: 'Dai Stablecoin', decimals: 18 }
]

const FEE_TIERS = [
  { value: 100, label: '0.01%', description: 'Stablecoin pairs' },
  { value: 500, label: '0.05%', description: 'Stable pairs' },
  { value: 3000, label: '0.30%', description: 'Most pairs' },
  { value: 10000, label: '1.00%', description: 'Exotic pairs' }
]

export default function VaultFactory() {
  const { address, isConnected } = useAccount()
  const [currentStep, setCurrentStep] = useState(1)
  const [selectedTemplate, setSelectedTemplate] = useState<string>('')
  const [formData, setFormData] = useState({
    name: '',
    symbol: '',
    token0: '',
    token1: '',
    feeTier: 3000,
    initialDeposit0: '',
    initialDeposit1: '',
    enableWhitelist: false,
    createPool: false
  })
  const [vaultConfig, setVaultConfig] = useState<VaultConfig>({
    strategyMode: 'AI_ASSISTED',
    positionType: 'DUAL_POSITION',
    autoCompound: true,
    rebalanceThreshold: 10,
    maxSlippage: 1
  })
  const [isDeploying, setIsDeploying] = useState(false)
  const [deploymentHash, setDeploymentHash] = useState<string>()

  const selectedTemplateData = VAULT_TEMPLATES.find(t => t.id === selectedTemplate)

  const handleTemplateSelect = (templateId: string) => {
    setSelectedTemplate(templateId)
    const template = VAULT_TEMPLATES.find(t => t.id === templateId)
    if (template) {
      // Update vault config based on template
      if (template.id === 'basic') {
        setVaultConfig({
          strategyMode: 'MANUAL',
          positionType: 'SINGLE_RANGE',
          autoCompound: true,
          rebalanceThreshold: 15,
          maxSlippage: 1
        })
      } else if (template.id === 'gamma') {
        setVaultConfig({
          strategyMode: 'AI_ASSISTED',
          positionType: 'DUAL_POSITION',
          autoCompound: true,
          rebalanceThreshold: 10,
          maxSlippage: 1
        })
      } else if (template.id === 'ai_optimized') {
        setVaultConfig({
          strategyMode: 'FULLY_AUTOMATED',
          positionType: 'AI_OPTIMIZED',
          autoCompound: true,
          rebalanceThreshold: 5,
          maxSlippage: 3
        })
      } else if (template.id === 'institutional') {
        setVaultConfig({
          strategyMode: 'AI_ASSISTED',
          positionType: 'MULTI_RANGE',
          autoCompound: true,
          rebalanceThreshold: 8,
          maxSlippage: 0.5
        })
      }
    }
  }

  const handleDeploy = async () => {
    if (!selectedTemplateData || !isConnected) return

    setIsDeploying(true)
    
    try {
      // Mock deployment for demo
      console.log('Deploying vault with config:', {
        template: selectedTemplate,
        formData,
        vaultConfig
      })
      // Simulate deployment delay
      setTimeout(() => {
        setIsDeploying(false)
        alert('Vault deployed successfully! (This is a demo)')
      }, 2000)
    } catch (error) {
      console.error('Deployment failed:', error)
      setIsDeploying(false)
    }
  }

  const canProceedToNext = () => {
    switch (currentStep) {
      case 1: return selectedTemplate !== ''
      case 2: return formData.name && formData.symbol && formData.token0 && formData.token1
      case 3: return true
      default: return false
    }
  }

  const getStepStatus = (step: number) => {
    if (step < currentStep) return 'completed'
    if (step === currentStep) return 'current'
    return 'upcoming'
  }

  const deploymentFee = 0.01 // ETH

  if (!isConnected) {
    return (
      <div className="min-h-screen bg-gray-50 dark:bg-gray-900 p-6 flex items-center justify-center">
        <div className="w-full max-w-md bg-white dark:bg-black border-2 border-black dark:border-white shadow-brutal dark:shadow-brutal p-6 text-center">
          <Factory className="w-12 h-12 text-gray-400 mx-auto mb-4" />
          <h2 className="text-xl font-semibold mb-2 text-gray-900 dark:text-white">Connect Wallet</h2>
          <p className="text-gray-600 dark:text-gray-400 mb-4">
            Please connect your wallet to create a new vault
          </p>
        </div>
      </div>
    )
  }

  return (
    <div className="min-h-screen bg-gray-50 dark:bg-gray-900 p-6">
      <div className="max-w-4xl mx-auto space-y-6">
        {/* Header */}
        <div className="text-center">
          <h1 className="text-3xl font-bold text-gray-900 dark:text-white mb-2">Create New Vault</h1>
          <p className="text-gray-600 dark:text-gray-400">
            Deploy a new automated liquidity management vault with AI optimization
          </p>
        </div>

        {/* Progress Steps */}
        <div className="flex items-center justify-center space-x-4 mb-8">
          {[
            { number: 1, title: 'Template' },
            { number: 2, title: 'Configuration' },
            { number: 3, title: 'Settings' },
            { number: 4, title: 'Deploy' }
          ].map((step, index) => (
            <div key={step.number} className="flex items-center">
              <div className={`
                flex items-center justify-center w-10 h-10 rounded-full border-2 font-medium
                ${getStepStatus(step.number) === 'completed' ? 'bg-green-500 border-green-500 text-white' :
                  getStepStatus(step.number) === 'current' ? 'bg-blue-500 border-blue-500 text-white' :
                  'bg-gray-200 border-gray-300 text-gray-500'}
              `}>
                {getStepStatus(step.number) === 'completed' ? (
                  <CheckCircle className="w-5 h-5" />
                ) : (
                  step.number
                )}
              </div>
              <span className="ml-2 text-sm font-medium text-gray-700 dark:text-gray-300">{step.title}</span>
              {index < 3 && <ChevronRight className="w-4 h-4 text-gray-400 mx-4" />}
            </div>
          ))}
        </div>

        {/* Step 1: Template Selection */}
        {currentStep === 1 && (
          <div className="bg-white dark:bg-black border-2 border-black dark:border-white shadow-brutal dark:shadow-brutal p-6">
            <h2 className="text-xl font-bold text-gray-900 dark:text-white mb-6">Choose Vault Template</h2>
            <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
              {VAULT_TEMPLATES.map((template) => (
                <div
                  key={template.id}
                  className={`
                    border-2 rounded-lg p-6 cursor-pointer transition-all
                    ${selectedTemplate === template.id ? 'border-blue-500 bg-blue-50 dark:bg-blue-900' : 'border-gray-300 hover:border-gray-400'}
                  `}
                  onClick={() => handleTemplateSelect(template.id)}
                >
                  <div className="flex items-center justify-between mb-4">
                    <div className="flex items-center space-x-3">
                      <div className={`
                        p-2 rounded-lg
                        ${selectedTemplate === template.id ? 'bg-blue-100 text-blue-600' : 'bg-gray-100 text-gray-600'}
                      `}>
                        {template.icon}
                      </div>
                      <div>
                        <h3 className="font-semibold text-gray-900 dark:text-white">{template.name}</h3>
                        <div className="flex items-center space-x-2 mt-1">
                          <span className={`inline-flex items-center px-2 py-1 rounded text-xs ${
                            template.riskLevel === 'Low' ? 'bg-green-100 text-green-800' : 
                            template.riskLevel === 'Medium' ? 'bg-yellow-100 text-yellow-800' : 
                            'bg-red-100 text-red-800'
                          }`}>
                            {template.riskLevel} Risk
                          </span>
                          {template.aiEnabled && (
                            <span className="inline-flex items-center px-2 py-1 rounded border border-purple-500 text-purple-700 text-xs">
                              <Sparkles className="w-3 h-3 mr-1" />
                              AI
                            </span>
                          )}
                        </div>
                      </div>
                    </div>
                  </div>
                  
                  <p className="text-sm text-gray-600 dark:text-gray-400 mb-4">{template.description}</p>
                  
                  <div className="space-y-2 mb-4">
                    {template.features.map((feature, index) => (
                      <div key={index} className="flex items-center text-sm text-gray-600 dark:text-gray-400">
                        <CheckCircle className="w-4 h-4 text-green-500 mr-2" />
                        {feature}
                      </div>
                    ))}
                  </div>
                  
                  <div className="grid grid-cols-3 gap-2 text-xs text-gray-500 dark:text-gray-400">
                    <div>
                      <p className="font-medium">Min Deposit</p>
                      <p>${template.minDeposit}</p>
                    </div>
                    <div>
                      <p className="font-medium">Mgmt Fee</p>
                      <p>{template.managementFee}%</p>
                    </div>
                    <div>
                      <p className="font-medium">Perf Fee</p>
                      <p>{template.performanceFee}%</p>
                    </div>
                  </div>
                </div>
              ))}
            </div>
          </div>
        )}

        {/* Step 2: Configuration */}
        {currentStep === 2 && (
          <div className="bg-white dark:bg-black border-2 border-black dark:border-white shadow-brutal dark:shadow-brutal p-6">
            <h2 className="text-xl font-bold text-gray-900 dark:text-white mb-6">Vault Configuration</h2>
            <div className="space-y-6">
              {/* Basic Info */}
              <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                <div>
                  <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-2">Vault Name</label>
                  <input
                    type="text"
                    placeholder="e.g., ETH/USDC Optimized"
                    value={formData.name}
                    onChange={(e) => setFormData({...formData, name: e.target.value})}
                    className="w-full px-3 py-2 border-2 border-black dark:border-white bg-white dark:bg-gray-900 text-black dark:text-white"
                  />
                </div>
                <div>
                  <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-2">Vault Symbol</label>
                  <input
                    type="text"
                    placeholder="e.g., dETH-USDC"
                    value={formData.symbol}
                    onChange={(e) => setFormData({...formData, symbol: e.target.value})}
                    className="w-full px-3 py-2 border-2 border-black dark:border-white bg-white dark:bg-gray-900 text-black dark:text-white"
                  />
                </div>
              </div>

              {/* Token Selection */}
              <div className="space-y-4">
                <h3 className="font-medium text-gray-900 dark:text-white">Token Pair</h3>
                <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                  <div>
                    <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-2">Token 0</label>
                    <select 
                      value={formData.token0} 
                      onChange={(e) => setFormData({...formData, token0: e.target.value})}
                      className="w-full px-3 py-2 border-2 border-black dark:border-white bg-white dark:bg-gray-900 text-black dark:text-white"
                    >
                      <option value="">Select token</option>
                      {POPULAR_TOKENS.map((token) => (
                        <option key={token.address} value={token.address}>
                          {token.symbol} - {token.name}
                        </option>
                      ))}
                    </select>
                  </div>
                  <div>
                    <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-2">Token 1</label>
                    <select 
                      value={formData.token1} 
                      onChange={(e) => setFormData({...formData, token1: e.target.value})}
                      className="w-full px-3 py-2 border-2 border-black dark:border-white bg-white dark:bg-gray-900 text-black dark:text-white"
                    >
                      <option value="">Select token</option>
                      {POPULAR_TOKENS.map((token) => (
                        <option key={token.address} value={token.address}>
                          {token.symbol} - {token.name}
                        </option>
                      ))}
                    </select>
                  </div>
                </div>
              </div>

              {/* Fee Tier */}
              <div>
                <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-2">Fee Tier</label>
                <div className="grid grid-cols-2 md:grid-cols-4 gap-2 mt-2">
                  {FEE_TIERS.map((tier) => (
                    <div
                      key={tier.value}
                      className={`
                        border-2 rounded-lg p-3 cursor-pointer text-center transition-all
                        ${formData.feeTier === tier.value ? 'border-blue-500 bg-blue-50 dark:bg-blue-900' : 'border-gray-300 hover:border-gray-400'}
                      `}
                      onClick={() => setFormData({...formData, feeTier: tier.value})}
                    >
                      <p className="font-medium text-gray-900 dark:text-white">{tier.label}</p>
                      <p className="text-xs text-gray-600 dark:text-gray-400">{tier.description}</p>
                    </div>
                  ))}
                </div>
              </div>
            </div>
          </div>
        )}

        {/* Step 3: Strategy Settings */}
        {currentStep === 3 && (
          <div className="bg-white dark:bg-black border-2 border-black dark:border-white shadow-brutal dark:shadow-brutal p-6">
            <h2 className="text-xl font-bold text-gray-900 dark:text-white mb-6">Strategy Settings</h2>
            <div className="space-y-6">
              <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
                <div>
                  <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-2">Strategy Mode</label>
                  <select 
                    value={vaultConfig.strategyMode} 
                    onChange={(e) => setVaultConfig({...vaultConfig, strategyMode: e.target.value as any})}
                    className="w-full px-3 py-2 border-2 border-black dark:border-white bg-white dark:bg-gray-900 text-black dark:text-white"
                  >
                    <option value="MANUAL">Manual Control</option>
                    <option value="AI_ASSISTED">AI Assisted</option>
                    <option value="FULLY_AUTOMATED">Fully Automated</option>
                  </select>
                </div>

                <div>
                  <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-2">Position Type</label>
                  <select 
                    value={vaultConfig.positionType} 
                    onChange={(e) => setVaultConfig({...vaultConfig, positionType: e.target.value as any})}
                    className="w-full px-3 py-2 border-2 border-black dark:border-white bg-white dark:bg-gray-900 text-black dark:text-white"
                  >
                    <option value="SINGLE_RANGE">Single Range</option>
                    <option value="DUAL_POSITION">Dual Position</option>
                    <option value="MULTI_RANGE">Multi Range</option>
                    <option value="AI_OPTIMIZED">AI Optimized</option>
                  </select>
                </div>

                <div>
                  <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-2">Rebalance Threshold (%)</label>
                  <input
                    type="number"
                    value={vaultConfig.rebalanceThreshold}
                    onChange={(e) => setVaultConfig({...vaultConfig, rebalanceThreshold: parseInt(e.target.value)})}
                    className="w-full px-3 py-2 border-2 border-black dark:border-white bg-white dark:bg-gray-900 text-black dark:text-white"
                  />
                </div>

                <div>
                  <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-2">Max Slippage (%)</label>
                  <input
                    type="number"
                    step="0.1"
                    value={vaultConfig.maxSlippage}
                    onChange={(e) => setVaultConfig({...vaultConfig, maxSlippage: parseFloat(e.target.value)})}
                    className="w-full px-3 py-2 border-2 border-black dark:border-white bg-white dark:bg-gray-900 text-black dark:text-white"
                  />
                </div>
              </div>

              <div className="flex items-center justify-between">
                <div>
                  <label className="text-sm font-medium text-gray-700 dark:text-gray-300">Auto Compound</label>
                  <p className="text-sm text-gray-600 dark:text-gray-400">Automatically compound earned fees</p>
                </div>
                <input
                  type="checkbox"
                  checked={vaultConfig.autoCompound}
                  onChange={(e) => setVaultConfig({...vaultConfig, autoCompound: e.target.checked})}
                  className="w-5 h-5"
                />
              </div>
            </div>
          </div>
        )}

        {/* Step 4: Deploy */}
        {currentStep === 4 && (
          <div className="bg-white dark:bg-black border-2 border-black dark:border-white shadow-brutal dark:shadow-brutal p-6">
            <h2 className="text-xl font-bold text-gray-900 dark:text-white mb-6">Deploy Vault</h2>
            <div className="space-y-6">
              {/* Summary */}
              <div className="bg-gray-50 dark:bg-gray-800 rounded-lg p-6">
                <h3 className="font-medium mb-4 text-gray-900 dark:text-white">Deployment Summary</h3>
                <div className="grid grid-cols-1 md:grid-cols-2 gap-4 text-sm">
                  <div>
                    <p className="text-gray-600 dark:text-gray-400">Template</p>
                    <p className="font-medium text-gray-900 dark:text-white">{selectedTemplateData?.name}</p>
                  </div>
                  <div>
                    <p className="text-gray-600 dark:text-gray-400">Name</p>
                    <p className="font-medium text-gray-900 dark:text-white">{formData.name}</p>
                  </div>
                  <div>
                    <p className="text-gray-600 dark:text-gray-400">Token Pair</p>
                    <p className="font-medium text-gray-900 dark:text-white">
                      {POPULAR_TOKENS.find(t => t.address === formData.token0)?.symbol} / 
                      {POPULAR_TOKENS.find(t => t.address === formData.token1)?.symbol}
                    </p>
                  </div>
                  <div>
                    <p className="text-gray-600 dark:text-gray-400">Fee Tier</p>
                    <p className="font-medium text-gray-900 dark:text-white">{formData.feeTier / 10000}%</p>
                  </div>
                  <div>
                    <p className="text-gray-600 dark:text-gray-400">Strategy Mode</p>
                    <p className="font-medium text-gray-900 dark:text-white">{vaultConfig.strategyMode.replace('_', ' ')}</p>
                  </div>
                  <div>
                    <p className="text-gray-600 dark:text-gray-400">Deployment Fee</p>
                    <p className="font-medium text-gray-900 dark:text-white">{deploymentFee} ETH</p>
                  </div>
                </div>
              </div>

              {/* Warnings */}
              <div className="bg-yellow-50 dark:bg-yellow-900 border-l-4 border-yellow-400 p-4">
                <div className="flex">
                  <AlertTriangle className="h-5 w-5 text-yellow-400" />
                  <div className="ml-3">
                    <p className="text-sm text-yellow-800 dark:text-yellow-200">
                      Please review all settings carefully. Some configurations cannot be changed after deployment.
                      The deployment fee is non-refundable.
                    </p>
                  </div>
                </div>
              </div>

              {/* Deploy Button */}
              <button
                onClick={handleDeploy}
                disabled={isDeploying}
                className="w-full bg-primary text-black px-6 py-3 border-2 border-black font-bold hover:translate-x-[-2px] hover:translate-y-[-2px] hover:shadow-[6px_6px_0px_0px_#000000] transition-all duration-150 disabled:opacity-50"
              >
                {isDeploying ? 'Deploying...' : `Deploy Vault (${deploymentFee} ETH)`}
              </button>
            </div>
          </div>
        )}

        {/* Navigation */}
        <div className="flex justify-between">
          <button
            onClick={() => setCurrentStep(Math.max(1, currentStep - 1))}
            disabled={currentStep === 1}
            className="bg-white dark:bg-black text-black dark:text-white px-4 py-2 border-2 border-black dark:border-white font-bold hover:translate-x-[-2px] hover:translate-y-[-2px] hover:shadow-[6px_6px_0px_0px_#000000] dark:hover:shadow-[6px_6px_0px_0px_#FFFFFF] transition-all duration-150 disabled:opacity-50"
          >
            Previous
          </button>
          
          {currentStep < 4 ? (
            <button
              onClick={() => setCurrentStep(currentStep + 1)}
              disabled={!canProceedToNext()}
              className="bg-primary text-black px-4 py-2 border-2 border-black font-bold hover:translate-x-[-2px] hover:translate-y-[-2px] hover:shadow-[6px_6px_0px_0px_#000000] transition-all duration-150 disabled:opacity-50"
            >
              Next
            </button>
          ) : (
            <div /> // Placeholder for alignment
          )}
        </div>
      </div>
    </div>
  )
}