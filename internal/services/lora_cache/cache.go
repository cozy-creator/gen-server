package lora_cache

import (
	// "encoding/json"
	"fmt"
	"os"
	"path/filepath"
	// "sort"
	// "sync"
	"time"
	"crypto/sha256"
	"encoding/hex"
	"net/http"
	"strings"
	"regexp"
	"net/url"

	// "github.com/dgraph-io/badger/v4"
)

type LoRACache struct {
	// db        *badger.DB
	baseDir   string
	threshold float64
	// mu        sync.RWMutex
}

// type loraEvictionInfo struct {
// 	metadata        LoRAMetadata
// 	score           float64
// }

func NewLoRACache(config Config) (*LoRACache, error) {
	if err := os.MkdirAll(config.BaseDir, 0755); err != nil {
		return nil, fmt.Errorf("failed to create LoRA directory: %w", err)
	}

	// open badger db
	// dbPath := filepath.Join(config.BaseDir, "metadata")
	// opts := badger.DefaultOptions(dbPath)
	// db, err := badger.Open(opts)
	// if err != nil {
	// 	return nil, fmt.Errorf("failed to open badger db: %w", err)
	// }

	return &LoRACache{
		// db:        db,
		baseDir:   config.BaseDir,
		threshold: config.EvictionThreshold,
	}, nil
}

// func (c *LoRACache) Close() error {
// 	return c.db.Close()
// }



func (c *LoRACache) GetLoRAPath(modelURL string) (string, error) {
    client := &http.Client{
        CheckRedirect: func(req *http.Request, via []*http.Request) error {
            return http.ErrUseLastResponse
        },
        Timeout: 30 * time.Second,
	}

	req, err := http.NewRequest("GET", modelURL, nil)
    if err != nil {
        return "", err
    }

    resp, err := client.Do(req)
    if err != nil {
        return "", err
    }
    defer resp.Body.Close()

    if resp.StatusCode != http.StatusFound && resp.StatusCode != http.StatusMovedPermanently && resp.StatusCode != http.StatusTemporaryRedirect {
        return "", fmt.Errorf("expected redirect response, got status %d", resp.StatusCode)
    }

    location := resp.Header.Get("Location")
    if location == "" {
        return "", fmt.Errorf("no redirect location found")
    }

    redirectURL, err := url.Parse(location)
    if err != nil {
        return "", fmt.Errorf("failed to parse redirect location: %w", err)
    }

    var filename string
    queryParams := redirectURL.Query()
    if contentDisp := queryParams.Get("response-content-disposition"); contentDisp != "" {
        re := regexp.MustCompile(`filename="([^"]+)`)
        if matches := re.FindStringSubmatch(contentDisp); len(matches) > 1 {
            filename = matches[1]
        }
    }

    if filename == "" && redirectURL.Path != "" {
        filename = filepath.Base(redirectURL.Path)
    }

    // Ensure filename has .safetensors extension
    if !strings.HasSuffix(filename, ".safetensors") {
        filename = strings.TrimSuffix(filename, filepath.Ext(filename)) + ".safetensors"
    }

    // Hash URL for uniqueness
    h := sha256.New()
    h.Write([]byte(modelURL))
    urlHash := hex.EncodeToString(h.Sum(nil))[:8]

    return filepath.Join(c.baseDir, fmt.Sprintf("%s--%s", strings.TrimSuffix(filename, ".safetensors"), urlHash) + ".safetensors"), nil
}

// func (c *LoRACache) Add(url string, filePath string) error {
// 	c.mu.Lock()
// 	defer c.mu.Unlock()


// 	metadata := LoRAMetadata{
// 		URL: url,
// 		FilePath: filePath,
// 		LastUsed: time.Now(),
// 		UsageCount: 1,
// 	}

// 	// store metadata in db
// 	data, err := json.Marshal(metadata)
// 	if err != nil {
// 		return fmt.Errorf("failed to marshal metadata: %w", err)
// 	}

// 	fmt.Println("Done marshalling metadata")

// 	err = c.db.Update(func(txn *badger.Txn) error {
// 		return txn.Set([]byte(url), data)
// 	})

// 	fmt.Println("Done storing metadata in db")
// 	if err != nil {
// 		return fmt.Errorf("failed to store metadata in db: %w", err)
// 	}

// 	fmt.Println("Starting check and evict")

// 	return c.checkAndEvict()
// }

// func (c *LoRACache) Get(url string) (*LoRAMetadata, error) {
// 	c.mu.RLock()
// 	defer c.mu.RUnlock()

// 	var metadata LoRAMetadata
// 	err := c.db.View(func(txn *badger.Txn) error {
// 		item, err := txn.Get([]byte(url))
// 		if err != nil {
// 			return err
// 		}

// 		return item.Value(func(val []byte) error {
// 			return json.Unmarshal(val, &metadata)
// 		})
// 	})

// 	if err == badger.ErrKeyNotFound {
// 		return nil, nil
// 	}
// 	if err != nil {
// 		return nil, fmt.Errorf("failed to get metadata from db: %w", err)
// 	}

// 	// update last used time and usage count
// 	err = c.updateUsage(url, &metadata)
// 	if err != nil {
// 		return nil, fmt.Errorf("failed to update usage: %w", err)
// 	}

// 	return &metadata, nil
// }

// func (c *LoRACache) updateUsage(url string, metadata *LoRAMetadata) error {
// 	metadata.LastUsed = time.Now()
// 	metadata.UsageCount++

// 	data, err := json.Marshal(metadata)
// 	if err != nil {
// 		return fmt.Errorf("failed to marshal metadata: %w", err)
// 	}

// 	return c.db.Update(func(txn *badger.Txn) error {
// 		return txn.Set([]byte(url), data)
// 	})
// }

// func (c *LoRACache) checkAndEvict() error {
// 	diskUsagePercent, err := getDiskUsage(c.baseDir)
// 	fmt.Println("Disk usage percent", diskUsagePercent)
// 	if err != nil {
// 		fmt.Println("Failed to get disk usage", err)
// 		return fmt.Errorf("failed to get disk usage: %w", err)
// 	}

// 	if diskUsagePercent < c.threshold {
// 		return nil
// 	}

// 	loras, err := c.getAllMetadata()
// 	if err != nil {
// 		fmt.Println("Failed to get LoRA metadata", err)
// 		return fmt.Errorf("failed to get LoRA metadata: %w", err)
// 	}

// 	fmt.Println("Done getting LoRA metadata")

// 	fmt.Println("Loras", loras)

// 	fmt.Println("calculating eviction scores")

// 	// calculate eviction scores
// 	var evictionCandidates []loraEvictionInfo
// 	now := time.Now()

// 	for _, lora := range loras {
// 		// calculate time factor (higher for older files)
// 		timeFactor := now.Sub(lora.LastUsed).Hours()
// 		fmt.Println("Time factor", timeFactor)

// 		// calculate usage factor (lower for frequently used files)
// 		usageFactor := 1.0 / float64(lora.UsageCount + 1)
// 		fmt.Println("Usage factor", usageFactor)

// 		// combine factors into final score where higher score == more likely to be evicted
// 		score := timeFactor * usageFactor
// 		fmt.Println("Score", score)


// 		evictionCandidates = append(evictionCandidates, loraEvictionInfo{
// 			metadata: lora,
// 			score: score,
// 		})
// 	}

// 	fmt.Println("Done calculating eviction scores")

// 	// sort by eviction score (highest first)
// 	sort.Slice(evictionCandidates, func(i, j int) bool {
// 		return evictionCandidates[i].score > evictionCandidates[j].score
// 	})

// 	fmt.Println("Done sorting eviction candidates")

// 	// start evicting the loras wuth hughest scores until we're below threshold
// 	for _, candidate := range evictionCandidates {
// 		if err := c.evictLoRA(candidate.metadata.URL); err != nil {
// 			fmt.Println("Failed to evict LoRA", candidate.metadata.URL, err)
// 			return fmt.Errorf("failed to evict LoRA %s: %w", candidate.metadata.URL, err)
// 		}
// 		// check if we're below threshold
// 		diskUsagePercent, err := getDiskUsage(c.baseDir)
// 		if err != nil {
// 			fmt.Println("Failed to get disk usage", err)
// 			return fmt.Errorf("failed to get disk usage: %w", err)
// 		}
// 		fmt.Println("Disk usage percent", diskUsagePercent)
// 		if diskUsagePercent < c.threshold {
// 			break
// 		}
// 	}

// 	fmt.Println("Done evicting LoRAs")

// 	return nil
// }


// func (c *LoRACache) getAllMetadata() ([]LoRAMetadata, error) {
// 	var loras []LoRAMetadata

// 	err := c.db.View(func(txn *badger.Txn) error {
// 		it := txn.NewIterator(badger.DefaultIteratorOptions)
// 		defer it.Close()

// 		for it.Rewind(); it.Valid(); it.Next() {
// 			item := it.Item()
// 			var metadata LoRAMetadata
// 			err := item.Value(func(val []byte) error {
// 				return json.Unmarshal(val, &metadata)
// 			})
// 			if err != nil {
// 				fmt.Println("Failed to unmarshal metadata", err)
// 				return fmt.Errorf("failed to unmarshal metadata: %w", err)
// 			}
// 			loras = append(loras, metadata)
// 		}
// 		return nil
// 	})

// 	if err != nil {
// 		return nil, fmt.Errorf("failed to get all LoRA metadata: %w", err)
// 	}

// 	return loras, nil
// }


// func (c *LoRACache) evictLoRA(url string) error {

// 	var metadata LoRAMetadata
// 	err := c.db.View(func(txn *badger.Txn) error {
// 		item, err := txn.Get([]byte(url))
// 		if err == badger.ErrKeyNotFound {
//             return fmt.Errorf("LoRA not found in cache")
//         }
//         if err != nil {
//             return fmt.Errorf("db error: %w", err)
//         }


// 		return item.Value(func(val []byte) error {
// 			return json.Unmarshal(val, &metadata)
// 		})
// 	})
// 	if err != nil {
// 		fmt.Println("Failed to get metadata for sgvsdgv eviction", err)
// 		return fmt.Errorf("failed to get metadata for eviction: %w", err)
// 	}

// 	// remove file
// 	if metadata.FilePath != "" {
//         if err := os.Remove(metadata.FilePath); err != nil {
//             if !os.IsNotExist(err) {
//                 return fmt.Errorf("failed to remove file %s: %w", metadata.FilePath, err)
//             }
//         }
//     }

// 	// remove from db
// 	err = c.db.Update(func(txn *badger.Txn) error {
// 		return txn.Delete([]byte(url))
// 	})
// 	if err != nil {
// 		return fmt.Errorf("failed to delete metadata from db: %w", err)
// 	}

// 	return nil
// }
