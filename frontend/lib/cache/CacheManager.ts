/**
 * @title CacheManager
 * @notice Multi-layer caching system for maximum performance
 * @dev Memory + Redis + IndexedDB with intelligent cache invalidation
 */

import { Redis } from 'ioredis';

// Cache configuration types
interface CacheConfig {
  memoryTTL: number;
  redisTTL: number;
  indexedDBTTL: number;
  maxMemorySize: number;
}

interface CacheItem<T> {
  data: T;
  timestamp: number;
  ttl: number;
  size: number;
}

interface CacheMetrics {
  memoryHits: number;
  redisHits: number;
  indexedDBHits: number;
  misses: number;
  evictions: number;
}

/**
 * @notice Level 1: In-Memory Cache (L1)
 * @dev LRU cache with size limits and TTL
 */
class MemoryCache {
  private cache = new Map<string, CacheItem<any>>();
  private accessOrder = new Map<string, number>();
  private currentSize = 0;
  private accessCounter = 0;

  constructor(private maxSize: number) {}

  get<T>(key: string): T | null {
    const item = this.cache.get(key);
    if (!item) return null;

    // Check TTL
    if (Date.now() - item.timestamp > item.ttl) {
      this.delete(key);
      return null;
    }

    // Update access order
    this.accessOrder.set(key, ++this.accessCounter);
    return item.data;
  }

  set<T>(key: string, data: T, ttl: number): void {
    const size = this.estimateSize(data);
    
    // Remove existing item if present
    if (this.cache.has(key)) {
      this.delete(key);
    }

    // Evict items if necessary
    while (this.currentSize + size > this.maxSize && this.cache.size > 0) {
      this.evictLRU();
    }

    // Add new item
    this.cache.set(key, {
      data,
      timestamp: Date.now(),
      ttl,
      size
    });
    this.accessOrder.set(key, ++this.accessCounter);
    this.currentSize += size;
  }

  delete(key: string): boolean {
    const item = this.cache.get(key);
    if (!item) return false;

    this.cache.delete(key);
    this.accessOrder.delete(key);
    this.currentSize -= item.size;
    return true;
  }

  private evictLRU(): void {
    let oldestKey = '';
    let oldestAccess = Infinity;

    for (const [key, access] of this.accessOrder) {
      if (access < oldestAccess) {
        oldestAccess = access;
        oldestKey = key;
      }
    }

    if (oldestKey) {
      this.delete(oldestKey);
    }
  }

  private estimateSize(data: any): number {
    return JSON.stringify(data).length * 2; // Rough byte estimate
  }

  clear(): void {
    this.cache.clear();
    this.accessOrder.clear();
    this.currentSize = 0;
  }

  getStats() {
    return {
      size: this.cache.size,
      currentSize: this.currentSize,
      maxSize: this.maxSize
    };
  }
}

/**
 * @notice Level 2: Redis Cache (L2)
 * @dev Distributed cache for cross-session persistence
 */
class RedisCache {
  private redis: Redis | null = null;

  constructor() {
    if (typeof window === 'undefined') {
      // Server-side Redis connection
      this.redis = new Redis({
        host: process.env.REDIS_HOST || 'localhost',
        port: parseInt(process.env.REDIS_PORT || '6379'),
        password: process.env.REDIS_PASSWORD,
        retryDelayOnFailover: 100,
        maxRetriesPerRequest: 3
      });
    }
  }

  async get<T>(key: string): Promise<T | null> {
    if (!this.redis) return null;

    try {
      const data = await this.redis.get(key);
      return data ? JSON.parse(data) : null;
    } catch (error) {
      console.warn('Redis get error:', error);
      return null;
    }
  }

  async set<T>(key: string, data: T, ttl: number): Promise<void> {
    if (!this.redis) return;

    try {
      await this.redis.setex(key, Math.floor(ttl / 1000), JSON.stringify(data));
    } catch (error) {
      console.warn('Redis set error:', error);
    }
  }

  async delete(key: string): Promise<void> {
    if (!this.redis) return;

    try {
      await this.redis.del(key);
    } catch (error) {
      console.warn('Redis delete error:', error);
    }
  }

  async clear(pattern?: string): Promise<void> {
    if (!this.redis) return;

    try {
      const keys = await this.redis.keys(pattern || '*');
      if (keys.length > 0) {
        await this.redis.del(...keys);
      }
    } catch (error) {
      console.warn('Redis clear error:', error);
    }
  }
}

/**
 * @notice Level 3: IndexedDB Cache (L3)
 * @dev Browser persistent storage for offline capability
 */
class IndexedDBCache {
  private dbName = 'DexterCache';
  private version = 1;
  private storeName = 'cache';

  private async getDB(): Promise<IDBDatabase> {
    return new Promise((resolve, reject) => {
      const request = indexedDB.open(this.dbName, this.version);

      request.onerror = () => reject(request.error);
      request.onsuccess = () => resolve(request.result);

      request.onupgradeneeded = (event) => {
        const db = (event.target as IDBOpenDBRequest).result;
        if (!db.objectStoreNames.contains(this.storeName)) {
          const store = db.createObjectStore(this.storeName, { keyPath: 'key' });
          store.createIndex('timestamp', 'timestamp', { unique: false });
        }
      };
    });
  }

  async get<T>(key: string): Promise<T | null> {
    if (typeof window === 'undefined') return null;

    try {
      const db = await this.getDB();
      const transaction = db.transaction([this.storeName], 'readonly');
      const store = transaction.objectStore(this.storeName);
      const request = store.get(key);

      return new Promise((resolve) => {
        request.onsuccess = () => {
          const result = request.result;
          if (!result) {
            resolve(null);
            return;
          }

          // Check TTL
          if (Date.now() - result.timestamp > result.ttl) {
            this.delete(key);
            resolve(null);
            return;
          }

          resolve(result.data);
        };
        request.onerror = () => resolve(null);
      });
    } catch (error) {
      console.warn('IndexedDB get error:', error);
      return null;
    }
  }

  async set<T>(key: string, data: T, ttl: number): Promise<void> {
    if (typeof window === 'undefined') return;

    try {
      const db = await this.getDB();
      const transaction = db.transaction([this.storeName], 'readwrite');
      const store = transaction.objectStore(this.storeName);

      await new Promise<void>((resolve, reject) => {
        const request = store.put({
          key,
          data,
          timestamp: Date.now(),
          ttl
        });

        request.onsuccess = () => resolve();
        request.onerror = () => reject(request.error);
      });
    } catch (error) {
      console.warn('IndexedDB set error:', error);
    }
  }

  async delete(key: string): Promise<void> {
    if (typeof window === 'undefined') return;

    try {
      const db = await this.getDB();
      const transaction = db.transaction([this.storeName], 'readwrite');
      const store = transaction.objectStore(this.storeName);

      await new Promise<void>((resolve) => {
        const request = store.delete(key);
        request.onsuccess = () => resolve();
        request.onerror = () => resolve();
      });
    } catch (error) {
      console.warn('IndexedDB delete error:', error);
    }
  }

  async clear(): Promise<void> {
    if (typeof window === 'undefined') return;

    try {
      const db = await this.getDB();
      const transaction = db.transaction([this.storeName], 'readwrite');
      const store = transaction.objectStore(this.storeName);

      await new Promise<void>((resolve) => {
        const request = store.clear();
        request.onsuccess = () => resolve();
        request.onerror = () => resolve();
      });
    } catch (error) {
      console.warn('IndexedDB clear error:', error);
    }
  }

  async cleanup(): Promise<void> {
    if (typeof window === 'undefined') return;

    try {
      const db = await this.getDB();
      const transaction = db.transaction([this.storeName], 'readwrite');
      const store = transaction.objectStore(this.storeName);
      const index = store.index('timestamp');

      const cutoff = Date.now() - (24 * 60 * 60 * 1000); // 24 hours
      const range = IDBKeyRange.upperBound(cutoff);

      await new Promise<void>((resolve) => {
        const request = index.openCursor(range);
        request.onsuccess = (event) => {
          const cursor = (event.target as IDBRequest).result;
          if (cursor) {
            cursor.delete();
            cursor.continue();
          } else {
            resolve();
          }
        };
        request.onerror = () => resolve();
      });
    } catch (error) {
      console.warn('IndexedDB cleanup error:', error);
    }
  }
}

/**
 * @notice Multi-Layer Cache Manager
 * @dev Orchestrates L1/L2/L3 caches with intelligent fallbacks
 */
export class CacheManager {
  private memoryCache: MemoryCache;
  private redisCache: RedisCache;
  private indexedDBCache: IndexedDBCache;
  private metrics: CacheMetrics;
  private config: CacheConfig;

  constructor(config: Partial<CacheConfig> = {}) {
    this.config = {
      memoryTTL: 5 * 60 * 1000, // 5 minutes
      redisTTL: 60 * 60 * 1000, // 1 hour
      indexedDBTTL: 24 * 60 * 60 * 1000, // 24 hours
      maxMemorySize: 50 * 1024 * 1024, // 50MB
      ...config
    };

    this.memoryCache = new MemoryCache(this.config.maxMemorySize);
    this.redisCache = new RedisCache();
    this.indexedDBCache = new IndexedDBCache();

    this.metrics = {
      memoryHits: 0,
      redisHits: 0,
      indexedDBHits: 0,
      misses: 0,
      evictions: 0
    };

    // Cleanup IndexedDB periodically
    if (typeof window !== 'undefined') {
      setInterval(() => this.indexedDBCache.cleanup(), 60 * 60 * 1000); // 1 hour
    }
  }

  /**
   * @notice Get data from cache with L1->L2->L3 fallback
   */
  async get<T>(key: string): Promise<T | null> {
    // L1: Memory cache
    const memoryData = this.memoryCache.get<T>(key);
    if (memoryData !== null) {
      this.metrics.memoryHits++;
      return memoryData;
    }

    // L2: Redis cache
    const redisData = await this.redisCache.get<T>(key);
    if (redisData !== null) {
      this.metrics.redisHits++;
      // Populate L1 cache
      this.memoryCache.set(key, redisData, this.config.memoryTTL);
      return redisData;
    }

    // L3: IndexedDB cache
    const indexedDBData = await this.indexedDBCache.get<T>(key);
    if (indexedDBData !== null) {
      this.metrics.indexedDBHits++;
      // Populate L1 and L2 caches
      this.memoryCache.set(key, indexedDBData, this.config.memoryTTL);
      await this.redisCache.set(key, indexedDBData, this.config.redisTTL);
      return indexedDBData;
    }

    this.metrics.misses++;
    return null;
  }

  /**
   * @notice Set data in all cache layers
   */
  async set<T>(key: string, data: T): Promise<void> {
    // Set in all layers with appropriate TTLs
    this.memoryCache.set(key, data, this.config.memoryTTL);
    await this.redisCache.set(key, data, this.config.redisTTL);
    await this.indexedDBCache.set(key, data, this.config.indexedDBTTL);
  }

  /**
   * @notice Delete from all cache layers
   */
  async delete(key: string): Promise<void> {
    this.memoryCache.delete(key);
    await this.redisCache.delete(key);
    await this.indexedDBCache.delete(key);
  }

  /**
   * @notice Clear all caches
   */
  async clear(): Promise<void> {
    this.memoryCache.clear();
    await this.redisCache.clear();
    await this.indexedDBCache.clear();
  }

  /**
   * @notice Get cache metrics for monitoring
   */
  getMetrics(): CacheMetrics & { hitRate: number; memoryStats: any } {
    const totalRequests = this.metrics.memoryHits + this.metrics.redisHits + 
                         this.metrics.indexedDBHits + this.metrics.misses;
    const hits = this.metrics.memoryHits + this.metrics.redisHits + this.metrics.indexedDBHits;
    
    return {
      ...this.metrics,
      hitRate: totalRequests > 0 ? hits / totalRequests : 0,
      memoryStats: this.memoryCache.getStats()
    };
  }

  /**
   * @notice Reset metrics
   */
  resetMetrics(): void {
    this.metrics = {
      memoryHits: 0,
      redisHits: 0,
      indexedDBHits: 0,
      misses: 0,
      evictions: 0
    };
  }
}

// Singleton instance
let cacheManager: CacheManager | null = null;

export function getCacheManager(): CacheManager {
  if (!cacheManager) {
    cacheManager = new CacheManager();
  }
  return cacheManager;
}

// Cache key generators
export const cacheKeys = {
  position: (id: string) => `position:${id}`,
  vault: (address: string) => `vault:${address}`,
  pool: (address: string) => `pool:${address}`,
  price: (token: string) => `price:${token}`,
  analytics: (address: string, timeframe: string) => `analytics:${address}:${timeframe}`,
  ml: (model: string, params: string) => `ml:${model}:${params}`,
  user: (address: string) => `user:${address}`
};