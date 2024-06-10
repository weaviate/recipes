import argparse
import weaviate

def create_backup(client):
    try:
        result = client.backup.create(
            backup_id="rag-with-recipes-backup",
            backend="filesystem",
            include_collections=["WeaviateRecipesChunk", "MemGPTMemory"],
            wait_for_completion=True,
        )
        print(f"Backup created successfully. Result: {result}")
    except Exception as e:
        print(f"Error creating backup: {e}")

def restore_backup(client):
    try:
        result = client.backup.restore(
            backup_id="rag-with-recipes-backup",
            backend="filesystem",
            wait_for_completion=True,
        )
        print(f"Backup restored successfully. Result: {result}")
    except Exception as e:
        print(f"Error restoring backup: {e}")

def main():
    parser = argparse.ArgumentParser(description="Weaviate Backup Manager")
    parser.add_argument("--create", action="store_true", help="Create a backup")
    parser.add_argument("--restore", action="store_true", help="Restore a backup")

    args = parser.parse_args()

    client = weaviate.connect_to_local()

    if args.create:
        create_backup(client)
    elif args.restore:
        restore_backup(client)
    else:
        print("Please specify either --create or --restore.")

    client.close()

if __name__ == "__main__":
    main()