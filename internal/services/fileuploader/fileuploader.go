package fileuploader

import (
	"bytes"
	"fmt"
	"io"
	"net/http"
	"net/url"
	"strings"

	"github.com/cozy-creator/gen-server/internal/services/filestorage"
	"github.com/cozy-creator/gen-server/internal/utils/hashutil"
	"github.com/gammazero/workerpool"
)

type Uploader struct {
	wp          *workerpool.WorkerPool
	filestorage filestorage.FileStorage
}

func NewFileUploader(filestorage filestorage.FileStorage, maxWorkers int) *Uploader {
	wp := workerpool.New(maxWorkers)

	return &Uploader{
		wp:          wp,
		filestorage: filestorage,
	}
}

func (w *Uploader) Stop() {
	w.wp.Stop()
}

func (w *Uploader) Upload(file filestorage.FileInfo, response chan string) {
	upload := func() {
		w.upload(file, response)
	}

	w.wp.Submit(upload)
}

func (w *Uploader) UploadBytes(file []byte, extension string, isTemp bool, response chan string) {
	fileHash := hashutil.Blake3Hash(file)
	fileInfo := filestorage.FileInfo{
		Name:      fileHash,
		Extension: extension,
		Content:   file,
		IsTemp:    isTemp,
		Kind:      filestorage.FileKindBytes,
	}

	w.Upload(fileInfo, response)
}

func (w *Uploader) UploadBytesPresigned(file []byte, presignedURL string, response chan string) {
	parsedURL, err := url.Parse(presignedURL)
	if err != nil {
		fmt.Sprintf("Failed to parse presigned URL: %v", err)
		return
	}

	s3Key := strings.TrimPrefix(parsedURL.Path, "/")
	req, err := http.NewRequest(http.MethodPut, presignedURL, bytes.NewReader(file))
	if err != nil {
		response <- fmt.Sprintf("Failed to create request: %v", err)
		return
	}

	req.Header.Set("Content-Type", "image/png")
	client := &http.Client{}
	resp, err := client.Do(req)
	if err != nil {
		fmt.Sprintf("Failed to upload file: %v", err)
		return
	}
	defer resp.Body.Close()

	// Check if the status code is successful
	if resp.StatusCode != http.StatusOK && resp.StatusCode != http.StatusNoContent {
		body, _ := io.ReadAll(resp.Body)
		fmt.Sprintf("Failed with status code %d: %s", resp.StatusCode, string(body))
		return
	}

	response <- s3Key
}

func (w *Uploader) UploadReader(reader io.Reader, extension string, isTemp bool, response chan string) {
	file, err := io.ReadAll(reader)
	if err != nil {
		panic(err)
	}

	fileHash := hashutil.Blake3Hash(file)
	fileInfo := filestorage.FileInfo{
		Content:   file,
		IsTemp:    isTemp,
		Name:      fileHash,
		Extension: extension,
		Kind:      filestorage.FileKindBytes,
	}

	w.Upload(fileInfo, response)
}

func (w *Uploader) upload(file filestorage.FileInfo, response chan string) {
	if w.filestorage == nil {
		return
	}

	url, err := w.filestorage.Upload(file)
	if err != nil {
		return
	}

	response <- url
}
