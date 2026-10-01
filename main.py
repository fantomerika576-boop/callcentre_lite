"""Запуск: python main.py [шлях до CSV або ZIP]"""
import sys

from app.gui import main

if __name__ == "__main__":
    main(sys.argv[1:])
