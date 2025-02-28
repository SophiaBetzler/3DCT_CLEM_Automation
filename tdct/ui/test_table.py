from PyQt5.QtCore import Qt, QAbstractTableModel, pyqtSignal, QModelIndex
from PyQt5.QtWidgets import (QApplication, QMainWindow, QTableView, 
                           QHeaderView, QVBoxLayout, QWidget)
import pandas as pd
import numpy as np

class ReorderableTableView(QTableView):
    """Custom TableView that enables drag and drop for rows only"""
    def __init__(self):
        super().__init__()
        # Enable drag and drop
        self.setDragEnabled(True)
        self.setAcceptDrops(True)
        self.setDragDropMode(QTableView.InternalMove)
        self.setDragDropOverwriteMode(False)
        
        # Enable selection
        self.setSelectionBehavior(QTableView.SelectRows)
        self.setSelectionMode(QTableView.SingleSelection)
        
        # Set up headers
        self.horizontalHeader().setSectionsMovable(False)  # Disable column moving
        self.verticalHeader().setSectionsMovable(True)
        self.verticalHeader().setDragEnabled(True)
        self.verticalHeader().setDragDropMode(QHeaderView.InternalMove)
        
        # Make the vertical header (row numbers) visible and styled
        self.verticalHeader().setVisible(True)
        self.verticalHeader().setDefaultAlignment(Qt.AlignRight | Qt.AlignVCenter)
        
        # Stretch columns to fill the widget
        self.horizontalHeader().setStretchLastSection(True)

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.resize(800, 400)
        
        # Create sample DataFrame
        self.df = pd.DataFrame({
            'Name': ['John', 'Alice', 'Bob', 'Carol'],
            'Age': [25, 30, 35, 28],
            'Salary': [50000.0, 65000.0, 75000.0, 60000.0],
            'Department': ['HR', 'Engineering', 'Sales', 'Marketing']
        })
        
        # Create table view and model
        self.table = QTableView()
        self.model = PandasTableModel(self.df)
        self.table.setModel(self.model)
        
        # Connect signals
        self.model.dataChanged.connect(self.on_data_changed)
        self.table.verticalHeader().sectionMoved.connect(self.on_row_moved)
        
        # Create central widget and layout
        central_widget = QWidget()
        layout = QVBoxLayout(central_widget)
        layout.addWidget(self.table)
        self.setCentralWidget(central_widget)
        
        # Set column widths
        header = self.table.horizontalHeader()
        for column in range(self.model.columnCount()):
            header.setSectionResizeMode(column, QHeaderView.Interactive)

    def on_data_changed(self, df):
        print("DataFrame updated:")
        print(df)

    def on_row_moved(self, logical_index, old_visual_index, new_visual_index):
        self.model.moveRow(old_visual_index, new_visual_index)

if __name__ == '__main__':
    app = QApplication([])
    window = MainWindow()
    window.show()
    app.exec_()