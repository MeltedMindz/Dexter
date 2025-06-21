'use client'

import React, { useState, useEffect } from 'react'
import { useAccount, useWriteContract, useWaitForTransactionReceipt } from 'wagmi'
import { parseEther, formatEther } from 'viem'
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card'
import { Button } from '@/components/ui/button'
import { Input } from '@/components/ui/input'
import { Label } from '@/components/ui/label'
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from '@/components/ui/select'
import { Switch } from '@/components/ui/switch'
import { Badge } from '@/components/ui/badge'
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/ui/tabs'
import { Alert, AlertDescription } from '@/components/ui/alert'
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
    icon: <Settings className=\"w-6 h-6\" />
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
    icon: <TrendingUp className=\"w-6 h-6\" />
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
    icon: <Zap className=\"w-6 h-6\" />
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
    icon: <Shield className=\"w-6 h-6\" />
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

  const { writeContract } = useWriteContract()
  const { isLoading: isConfirming, isSuccess } = useWaitForTransactionReceipt({
    hash: deploymentHash as `0x${string}`
  })

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
      const deploymentParams = {
        token0: formData.token0,
        token1: formData.token1,
        fee: formData.feeTier,
        templateType: selectedTemplate,
        vaultConfig: vaultConfig,
        name: formData.name,
        symbol: formData.symbol,
        initialDeposit0: parseEther(formData.initialDeposit0 || '0'),
        initialDeposit1: parseEther(formData.initialDeposit1 || '0'),
        createPool: formData.createPool,
        enableWhitelist: formData.enableWhitelist,
        initialWhitelist: [],
        customData: '0x'
      }

      await writeContract({
        address: '0x...', // VaultFactory address
        abi: [], // VaultFactory ABI
        functionName: 'createVault',
        args: [deploymentParams],
        value: parseEther('0.01') // Deployment fee
      })
    } catch (error) {
      console.error('Deployment failed:', error)
    } finally {
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
      <div className=\"min-h-screen bg-gray-50 p-6 flex items-center justify-center\">
        <Card className=\"w-full max-w-md\">
          <CardContent className=\"p-6 text-center\">
            <Factory className=\"w-12 h-12 text-gray-400 mx-auto mb-4\" />
            <h2 className=\"text-xl font-semibold mb-2\">Connect Wallet</h2>
            <p className=\"text-gray-600 mb-4\">
              Please connect your wallet to create a new vault
            </p>
          </CardContent>
        </Card>
      </div>
    )
  }

  return (
    <div className=\"min-h-screen bg-gray-50 p-6\">
      <div className=\"max-w-4xl mx-auto space-y-6\">
        {/* Header */}
        <div className=\"text-center\">
          <h1 className=\"text-3xl font-bold text-gray-900 mb-2\">Create New Vault</h1>
          <p className=\"text-gray-600\">
            Deploy a new automated liquidity management vault with AI optimization
          </p>
        </div>

        {/* Progress Steps */}
        <div className=\"flex items-center justify-center space-x-4 mb-8\">
          {[
            { number: 1, title: 'Template' },
            { number: 2, title: 'Configuration' },
            { number: 3, title: 'Settings' },
            { number: 4, title: 'Deploy' }
          ].map((step, index) => (
            <div key={step.number} className=\"flex items-center\">
              <div className={`
                flex items-center justify-center w-10 h-10 rounded-full border-2 font-medium
                ${getStepStatus(step.number) === 'completed' ? 'bg-green-500 border-green-500 text-white' :
                  getStepStatus(step.number) === 'current' ? 'bg-blue-500 border-blue-500 text-white' :
                  'bg-gray-200 border-gray-300 text-gray-500'}
              `}>
                {getStepStatus(step.number) === 'completed' ? (
                  <CheckCircle className=\"w-5 h-5\" />
                ) : (
                  step.number
                )}
              </div>
              <span className=\"ml-2 text-sm font-medium text-gray-700\">{step.title}</span>
              {index < 3 && <ChevronRight className=\"w-4 h-4 text-gray-400 mx-4\" />}
            </div>
          ))}
        </div>

        {/* Step 1: Template Selection */}
        {currentStep === 1 && (
          <Card>
            <CardHeader>
              <CardTitle>Choose Vault Template</CardTitle>
            </CardHeader>
            <CardContent>
              <div className=\"grid grid-cols-1 md:grid-cols-2 gap-6\">
                {VAULT_TEMPLATES.map((template) => (
                  <div
                    key={template.id}
                    className={`
                      border rounded-lg p-6 cursor-pointer transition-all
                      ${selectedTemplate === template.id ? 'border-blue-500 bg-blue-50' : 'border-gray-200 hover:border-gray-300'}
                    `}
                    onClick={() => handleTemplateSelect(template.id)}
                  >
                    <div className=\"flex items-center justify-between mb-4\">
                      <div className=\"flex items-center space-x-3\">
                        <div className={`
                          p-2 rounded-lg
                          ${selectedTemplate === template.id ? 'bg-blue-100 text-blue-600' : 'bg-gray-100 text-gray-600'}
                        `}>
                          {template.icon}
                        </div>
                        <div>
                          <h3 className=\"font-semibold\">{template.name}</h3>
                          <div className=\"flex items-center space-x-2 mt-1\">
                            <Badge variant={template.riskLevel === 'Low' ? 'default' : template.riskLevel === 'Medium' ? 'secondary' : 'destructive'}>
                              {template.riskLevel} Risk
                            </Badge>
                            {template.aiEnabled && (
                              <Badge variant=\"outline\" className=\"bg-purple-50 text-purple-700\">
                                <Sparkles className=\"w-3 h-3 mr-1\" />
                                AI
                              </Badge>
                            )}
                          </div>
                        </div>
                      </div>
                    </div>
                    
                    <p className=\"text-sm text-gray-600 mb-4\">{template.description}</p>
                    
                    <div className=\"space-y-2 mb-4\">
                      {template.features.map((feature, index) => (
                        <div key={index} className=\"flex items-center text-sm text-gray-600\">
                          <CheckCircle className=\"w-4 h-4 text-green-500 mr-2\" />
                          {feature}
                        </div>
                      ))}
                    </div>
                    
                    <div className=\"grid grid-cols-3 gap-2 text-xs text-gray-500\">
                      <div>
                        <p className=\"font-medium\">Min Deposit</p>
                        <p>${template.minDeposit}</p>
                      </div>
                      <div>
                        <p className=\"font-medium\">Mgmt Fee</p>
                        <p>{template.managementFee}%</p>
                      </div>
                      <div>
                        <p className=\"font-medium\">Perf Fee</p>
                        <p>{template.performanceFee}%</p>
                      </div>
                    </div>
                  </div>
                ))}
              </div>
            </CardContent>
          </Card>
        )}

        {/* Step 2: Configuration */}
        {currentStep === 2 && (
          <Card>
            <CardHeader>
              <CardTitle>Vault Configuration</CardTitle>
            </CardHeader>
            <CardContent className=\"space-y-6\">
              {/* Basic Info */}
              <div className=\"grid grid-cols-1 md:grid-cols-2 gap-4\">
                <div>
                  <Label htmlFor=\"name\">Vault Name</Label>
                  <Input
                    id=\"name\"
                    placeholder=\"e.g., ETH/USDC Optimized\"
                    value={formData.name}
                    onChange={(e) => setFormData({...formData, name: e.target.value})}
                  />
                </div>
                <div>
                  <Label htmlFor=\"symbol\">Vault Symbol</Label>
                  <Input
                    id=\"symbol\"
                    placeholder=\"e.g., dETH-USDC\"
                    value={formData.symbol}
                    onChange={(e) => setFormData({...formData, symbol: e.target.value})}
                  />
                </div>
              </div>

              {/* Token Selection */}
              <div className=\"space-y-4\">
                <h3 className=\"font-medium\">Token Pair</h3>
                <div className=\"grid grid-cols-1 md:grid-cols-2 gap-4\">
                  <div>
                    <Label htmlFor=\"token0\">Token 0</Label>
                    <Select value={formData.token0} onValueChange={(value) => setFormData({...formData, token0: value})}>
                      <SelectTrigger>
                        <SelectValue placeholder=\"Select token\" />
                      </SelectTrigger>
                      <SelectContent>
                        {POPULAR_TOKENS.map((token) => (
                          <SelectItem key={token.address} value={token.address}>
                            {token.symbol} - {token.name}
                          </SelectItem>
                        ))}
                      </SelectContent>
                    </Select>
                  </div>
                  <div>
                    <Label htmlFor=\"token1\">Token 1</Label>
                    <Select value={formData.token1} onValueChange={(value) => setFormData({...formData, token1: value})}>
                      <SelectTrigger>
                        <SelectValue placeholder=\"Select token\" />
                      </SelectTrigger>
                      <SelectContent>
                        {POPULAR_TOKENS.map((token) => (
                          <SelectItem key={token.address} value={token.address}>
                            {token.symbol} - {token.name}
                          </SelectItem>
                        ))}
                      </SelectContent>
                    </Select>
                  </div>
                </div>
              </div>

              {/* Fee Tier */}
              <div>
                <Label>Fee Tier</Label>
                <div className=\"grid grid-cols-2 md:grid-cols-4 gap-2 mt-2\">
                  {FEE_TIERS.map((tier) => (
                    <div
                      key={tier.value}
                      className={`
                        border rounded-lg p-3 cursor-pointer text-center transition-all
                        ${formData.feeTier === tier.value ? 'border-blue-500 bg-blue-50' : 'border-gray-200 hover:border-gray-300'}
                      `}
                      onClick={() => setFormData({...formData, feeTier: tier.value})}
                    >
                      <p className=\"font-medium\">{tier.label}</p>
                      <p className=\"text-xs text-gray-600\">{tier.description}</p>
                    </div>
                  ))}
                </div>
              </div>

              {/* Initial Liquidity */}
              <div className=\"space-y-4\">
                <h3 className=\"font-medium\">Initial Liquidity (Optional)</h3>
                <div className=\"grid grid-cols-1 md:grid-cols-2 gap-4\">
                  <div>
                    <Label htmlFor=\"deposit0\">Token 0 Amount</Label>
                    <Input
                      id=\"deposit0\"
                      type=\"number\"
                      placeholder=\"0.0\"
                      value={formData.initialDeposit0}
                      onChange={(e) => setFormData({...formData, initialDeposit0: e.target.value})}
                    />
                  </div>
                  <div>
                    <Label htmlFor=\"deposit1\">Token 1 Amount</Label>
                    <Input
                      id=\"deposit1\"
                      type=\"number\"
                      placeholder=\"0.0\"
                      value={formData.initialDeposit1}
                      onChange={(e) => setFormData({...formData, initialDeposit1: e.target.value})}
                    />
                  </div>
                </div>
              </div>

              {/* Advanced Options */}
              <div className=\"space-y-4\">
                <h3 className=\"font-medium\">Advanced Options</h3>
                <div className=\"space-y-3\">
                  <div className=\"flex items-center justify-between\">
                    <div>
                      <Label>Create Pool if Not Exists</Label>
                      <p className=\"text-sm text-gray-600\">Automatically create Uniswap V3 pool</p>
                    </div>
                    <Switch
                      checked={formData.createPool}
                      onCheckedChange={(checked) => setFormData({...formData, createPool: checked})}
                    />
                  </div>
                  <div className=\"flex items-center justify-between\">
                    <div>
                      <Label>Enable Whitelist</Label>
                      <p className=\"text-sm text-gray-600\">Restrict deposits to whitelisted addresses</p>
                    </div>
                    <Switch
                      checked={formData.enableWhitelist}
                      onCheckedChange={(checked) => setFormData({...formData, enableWhitelist: checked})}
                    />
                  </div>
                </div>
              </div>
            </CardContent>
          </Card>
        )}

        {/* Step 3: Strategy Settings */}
        {currentStep === 3 && (
          <Card>
            <CardHeader>
              <CardTitle>Strategy Settings</CardTitle>
            </CardHeader>
            <CardContent className=\"space-y-6\">
              <div className=\"grid grid-cols-1 md:grid-cols-2 gap-6\">
                <div>
                  <Label>Strategy Mode</Label>
                  <Select 
                    value={vaultConfig.strategyMode} 
                    onValueChange={(value: any) => setVaultConfig({...vaultConfig, strategyMode: value})}
                  >
                    <SelectTrigger>
                      <SelectValue />
                    </SelectTrigger>
                    <SelectContent>
                      <SelectItem value=\"MANUAL\">Manual Control</SelectItem>
                      <SelectItem value=\"AI_ASSISTED\">AI Assisted</SelectItem>
                      <SelectItem value=\"FULLY_AUTOMATED\">Fully Automated</SelectItem>
                    </SelectContent>
                  </Select>
                </div>

                <div>
                  <Label>Position Type</Label>
                  <Select 
                    value={vaultConfig.positionType} 
                    onValueChange={(value: any) => setVaultConfig({...vaultConfig, positionType: value})}
                  >
                    <SelectTrigger>
                      <SelectValue />
                    </SelectTrigger>
                    <SelectContent>
                      <SelectItem value=\"SINGLE_RANGE\">Single Range</SelectItem>
                      <SelectItem value=\"DUAL_POSITION\">Dual Position</SelectItem>
                      <SelectItem value=\"MULTI_RANGE\">Multi Range</SelectItem>
                      <SelectItem value=\"AI_OPTIMIZED\">AI Optimized</SelectItem>
                    </SelectContent>
                  </Select>
                </div>

                <div>
                  <Label>Rebalance Threshold (%)</Label>
                  <Input
                    type=\"number\"
                    value={vaultConfig.rebalanceThreshold}
                    onChange={(e) => setVaultConfig({...vaultConfig, rebalanceThreshold: parseInt(e.target.value)})}
                  />
                </div>

                <div>
                  <Label>Max Slippage (%)</Label>
                  <Input
                    type=\"number\"
                    step=\"0.1\"
                    value={vaultConfig.maxSlippage}
                    onChange={(e) => setVaultConfig({...vaultConfig, maxSlippage: parseFloat(e.target.value)})}
                  />
                </div>
              </div>

              <div className=\"flex items-center justify-between\">
                <div>
                  <Label>Auto Compound</Label>
                  <p className=\"text-sm text-gray-600\">Automatically compound earned fees</p>
                </div>
                <Switch
                  checked={vaultConfig.autoCompound}
                  onCheckedChange={(checked) => setVaultConfig({...vaultConfig, autoCompound: checked})}
                />
              </div>
            </CardContent>
          </Card>
        )}

        {/* Step 4: Deploy */}
        {currentStep === 4 && (
          <Card>
            <CardHeader>
              <CardTitle>Deploy Vault</CardTitle>
            </CardHeader>
            <CardContent className=\"space-y-6\">
              {/* Summary */}
              <div className=\"bg-gray-50 rounded-lg p-6\">
                <h3 className=\"font-medium mb-4\">Deployment Summary</h3>
                <div className=\"grid grid-cols-1 md:grid-cols-2 gap-4 text-sm\">
                  <div>
                    <p className=\"text-gray-600\">Template</p>
                    <p className=\"font-medium\">{selectedTemplateData?.name}</p>
                  </div>
                  <div>
                    <p className=\"text-gray-600\">Name</p>
                    <p className=\"font-medium\">{formData.name}</p>
                  </div>
                  <div>
                    <p className=\"text-gray-600\">Token Pair</p>
                    <p className=\"font-medium\">
                      {POPULAR_TOKENS.find(t => t.address === formData.token0)?.symbol} / 
                      {POPULAR_TOKENS.find(t => t.address === formData.token1)?.symbol}
                    </p>
                  </div>
                  <div>
                    <p className=\"text-gray-600\">Fee Tier</p>
                    <p className=\"font-medium\">{formData.feeTier / 10000}%</p>
                  </div>
                  <div>
                    <p className=\"text-gray-600\">Strategy Mode</p>
                    <p className=\"font-medium\">{vaultConfig.strategyMode.replace('_', ' ')}</p>
                  </div>
                  <div>
                    <p className=\"text-gray-600\">Deployment Fee</p>
                    <p className=\"font-medium\">{deploymentFee} ETH</p>
                  </div>
                </div>
              </div>

              {/* Warnings */}
              <Alert>
                <AlertTriangle className=\"h-4 w-4\" />
                <AlertDescription>
                  Please review all settings carefully. Some configurations cannot be changed after deployment.
                  The deployment fee is non-refundable.
                </AlertDescription>
              </Alert>

              {/* Deploy Button */}
              <Button
                onClick={handleDeploy}
                disabled={isDeploying || isConfirming}
                size=\"lg\"
                className=\"w-full\"
              >
                {isDeploying ? 'Deploying...' : isConfirming ? 'Confirming...' : `Deploy Vault (${deploymentFee} ETH)`}
              </Button>

              {/* Success */}
              {isSuccess && (
                <Alert className=\"border-green-200 bg-green-50\">
                  <CheckCircle className=\"h-4 w-4 text-green-600\" />
                  <AlertDescription className=\"text-green-800\">
                    Vault deployed successfully! You can now manage your vault from the dashboard.
                  </AlertDescription>
                </Alert>
              )}
            </CardContent>
          </Card>
        )}

        {/* Navigation */}
        <div className=\"flex justify-between\">
          <Button
            variant=\"outline\"
            onClick={() => setCurrentStep(Math.max(1, currentStep - 1))}
            disabled={currentStep === 1}
          >
            Previous
          </Button>
          
          {currentStep < 4 ? (
            <Button
              onClick={() => setCurrentStep(currentStep + 1)}
              disabled={!canProceedToNext()}
            >
              Next
            </Button>
          ) : (
            <div /> // Placeholder for alignment
          )}
        </div>
      </div>
    </div>
  )
}