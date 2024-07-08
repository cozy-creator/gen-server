# import argparse
#
#
# def main():
#     parser = argparse.ArgumentParser(description="Cozy Creator")
#     subparsers = parser.add_subparsers(dest="command", help="Sub-command help")
#
#     # add 'run' command
#     run_parser = subparsers.add_parser("run", help="Run the Cozy Gen Server")
#     run_parser.add_argument("--env-file", help="Path to .env file")
#     run_parser.add_argument("--secrets-dir", help="Path to secrets directory")
#
#     # print(run_parser)
#
#     args = parser.parse_args()
#
#     if args.command == "run":
#         print("runnn")
#
#
# if __name__ == "__main__":
#     main()
