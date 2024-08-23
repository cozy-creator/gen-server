package cmd

import (
	"fmt"

	"github.com/cozy-creator/hf-hub/hub"
	"github.com/spf13/cobra"
)

var downloadCmd = &cobra.Command{
	Use:   "download",
	Short: "Download models from hugging face",
	RunE: func(cmd *cobra.Command, args []string) error {
		repoId, err := cmd.Flags().GetString("repo-id")
		if err != nil {
			return err
		}

		fileName, err := cmd.Flags().GetString("file-name")
		if err != nil {
			return err
		}

		subFolder, err := cmd.Flags().GetString("sub-folder")
		if err != nil {
			return err
		}

		repoType, err := cmd.Flags().GetString("repo-type")
		if err != nil {
			return err
		}

		forceDownload, err := cmd.Flags().GetBool("force-download")
		if err != nil {
			return err
		}

		cacheDir, err := cmd.Flags().GetString("cache-dir")
		if err != nil {
			return err
		}

		revision, err := cmd.Flags().GetString("revision")
		if err != nil {
			return err
		}

		client := hub.DefaultClient()
		if cacheDir != "" {
			client.WithCacheDir(cacheDir)
		}

		repo := hub.NewHfRepo(repoId).WithType(repoType)

		if revision != "" {
			repo = repo.WithRevision(revision)
		}

		var filePath string
		if fileName != "" {
			file := repo.File(fileName).WithSubFolder(subFolder)
			filePath, err = client.FileDownload(file, forceDownload, false)
		} else {
			filePath, err = client.SnapshotDownload(repo, forceDownload, false)
		}

		if err != nil {
			return err
		}

		fmt.Println("Download complete: ", filePath)
		return nil
	},
}

func init() {
	downloadCmd.Flags().String("cache-dir", "", "The directory to cache the downloaded file or repo")
	downloadCmd.Flags().String("repo-id", "", "The ID of the model repository to download from")
	downloadCmd.Flags().String("file-name", "", "The name of the file to download, if not specified, the entire repo will be downloaded")
	downloadCmd.Flags().String("sub-folder", "", "The subfolder within the repo to download from")
	downloadCmd.Flags().String("variant", "", "The variant of the model to download")
	downloadCmd.Flags().String("repo-type", "", "The type of the repo to download")
	downloadCmd.Flags().Bool("force-download", false, "Force download of the file or repo")
	downloadCmd.Flags().String("revision", "", "The revision of the model to download")
}