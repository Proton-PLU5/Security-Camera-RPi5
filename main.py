from data.storage import StorageThread

if __name__ == "__main__":
    storage_thread = StorageThread()
    storage_thread.start()
    
    while True:
        user_input = input("> ")
        if user_input.lower() == "exit":
            break
    
    storage_thread.join()