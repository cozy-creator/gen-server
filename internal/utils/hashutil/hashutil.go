package hashutil

import (
	"encoding/hex"

	"lukechampine.com/blake3"
)

func Blake3Hash(data []byte) string {
	hash := blake3.Sum256(data)
	return hex.EncodeToString(hash[:])
}
