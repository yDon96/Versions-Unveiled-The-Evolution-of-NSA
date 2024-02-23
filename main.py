from app.app import run

if __name__ == '__main__':
    try:
        run()
    except KeyboardInterrupt:
        print(f'Execution stopped for a CTRL+C interrupt.')
