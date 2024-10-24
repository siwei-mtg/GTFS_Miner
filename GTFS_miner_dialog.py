# -*- coding: utf-8 -*-
"""
/***************************************************************************
 GTFS_minerDialog
                                 A QGIS plugin
 Extraction facile des données GTFS
 Generated by Plugin Builder: http://g-sherman.github.io/Qgis-Plugin-Builder/
                             -------------------
        begin                : 2020-11-08
        git sha              : $Format:%H$
        copyright            : (C) 2020 by Wei SI Transamo
        email                : wei.si@transamo.com
 ***************************************************************************/

/***************************************************************************
 *                                                                         *
 *   This program is free software; you can redistribute it and/or modify  *
 *   it under the terms of the GNU General Public License as published by  *
 *   the Free Software Foundation; either version 2 of the License, or     *
 *   (at your option) any later version.                                   *
 *                                                                         *
 ***************************************************************************/
"""

import os
import pandas as pd
from qgis.PyQt import uic
from qgis.PyQt import QtWidgets
from PyQt5.QtGui import QStandardItemModel, QStandardItem


# This loads your .ui file so that PyQt can populate your plugin with the elements from Qt Designer
FORM_CLASS, _ = uic.loadUiType(os.path.join(
    os.path.dirname(__file__), 'GTFS_miner_dialog_base.ui'))


class GTFS_minerDialog(QtWidgets.QDialog, FORM_CLASS):
    def __init__(self, parent=None):
        """Constructor."""
        super(GTFS_minerDialog, self).__init__(parent)
        # Set up the user interface from Designer through FORM_CLASS.
        # After self.setupUi() you can access any designer object by doing
        # self.<objectname>, and you can use autoconnect slots - see
        # http://qt-project.org/doc/qt-4.8/designer-using-a-ui-file.html
        # #widgets-and-dialogs-with-auto-connect
        self.setupUi(self)
        # Create an instance of MatplotlibWidget

    def initDialog(self):
        self.comboBox_zonevac.addItems(['Type_Jour_Vacances_A','Type_Jour_Vacances_B','Type_Jour_Vacances_C'])
        self.progressBar.setMinimum(1)
        self.progressBar.setMaximum(100)
        self.progressBar.setValue(1)
        self.datasetQualimodel = QStandardItemModel()  
        self.tableQualimodel = QStandardItemModel()  
        self.test1Model = QStandardItemModel()  
        self.test2Model = QStandardItemModel()  
        self.test3Model = QStandardItemModel()  
        self.test4Model = QStandardItemModel()  


    def connectSignals(self, user_input_dir, user_output_dir, exec_func):
        """Connect dialog buttons to functions."""
        self.pushButton_input.clicked.connect(user_input_dir)
        self.pushButton_output.clicked.connect(user_output_dir)
        self.execute.clicked.connect(exec_func)
        self.closeButton.clicked.connect(self.onClose)
        self.closeButton2.clicked.connect(self.onClose)  # Connect Close button

    def onClose(self):
        self.resetFields()  # Clear the fields
        self.reject()  # Close the dialog with a result of False

    def resetFields(self):
        """Reset dialog fields for the next run."""
        self.lineEdit_input.clear()
        self.lineEdit_output.clear()
        self.progressText.clear()
        self.comboBox_zonevac.clear()
        self.datasetQualimodel.clear()
        self.tableQualimodel.clear()
        self.test1Model.clear()
        self.test2Model.clear()
        self.test3Model.clear()
        self.test4Model.clear()

    def disconnectSignals(self):
        """Disconnect dialog buttons to avoid issues with memory."""
        self.pushButton_input.clicked.disconnect()
        self.pushButton_output.clicked.disconnect()
        self.execute.clicked.disconnect()
    
    def datasetQualityTableView(self, df):
        """Populate the QTableView with output data (expected as a pandas DataFrame)."""
        # Clear the existing data in the model
        self.datasetQualimodel.clear()
        # Ensure df is a DataFrame; if not, convert or handle appropriately
        if not isinstance(df, pd.DataFrame):
            logger.info("Data is not a DataFrame.")
            return
        headers = df.columns.tolist()
        # Set headers
        self.datasetQualimodel.setHorizontalHeaderLabels(headers)
        # Populate the model with data
        for index, row_data in df.iterrows():
            row = []
            for item in row_data:
                row.append(QStandardItem(str(item)))  # Ensure everything is converted to string
            self.datasetQualimodel.appendRow(row)
        # Update the table view
        self.datasetLevelInfo.setModel(self.datasetQualimodel)

    def tableQualityTableView(self, df):
        """Populate the QTableView with output data (expected as a pandas DataFrame)."""
        # Clear the existing data in the model
        self.tableQualimodel.clear()

        # Ensure df is a DataFrame; if not, convert or handle appropriately
        if not isinstance(df, pd.DataFrame):
            logger.info("Data is not a DataFrame.")
            return

        headers = df.columns.tolist()

        # Set headers
        self.tableQualimodel.setHorizontalHeaderLabels(headers)

        # Populate the model with data
        for index, row_data in df.iterrows():
            row = []
            for item in row_data:
                row.append(QStandardItem(str(item)))  # Ensure everything is converted to string
            self.tableQualimodel.appendRow(row)

        # Update the table view
        self.tableLevelInfo.setModel(self.tableQualimodel)

    def test1TableView(self, df):
        """Populate the QTableView with output data (expected as a pandas DataFrame)."""
        # Clear the existing data in the model
        self.test1Model.clear()

        # Ensure df is a DataFrame; if not, convert or handle appropriately
        if not isinstance(df, pd.DataFrame):
            logger.info("Data is not a DataFrame.")
            return

        headers = df.columns.tolist()

        # Set headers
        self.test1Model.setHorizontalHeaderLabels(headers)

        # Populate the model with data
        for index, row_data in df.iterrows():
            row = []
            for item in row_data:
                row.append(QStandardItem(str(item)))  # Ensure everything is converted to string
            self.test1Model.appendRow(row)

        # Update the table view
        self.tableView_1.setModel(self.test1Model)

    def test2TableView(self, df):
        """Populate the QTableView with output data (expected as a pandas DataFrame)."""
        # Clear the existing data in the model
        self.test2Model.clear()

        # Ensure df is a DataFrame; if not, convert or handle appropriately
        if not isinstance(df, pd.DataFrame):
            logger.info("Data is not a DataFrame.")
            return

        headers = df.columns.tolist()

        # Set headers
        self.test2Model.setHorizontalHeaderLabels(headers)

        # Populate the model with data
        for index, row_data in df.iterrows():
            row = []
            for item in row_data:
                row.append(QStandardItem(str(item)))  # Ensure everything is converted to string
            self.test2Model.appendRow(row)

        # Update the table view
        self.tableView_2.setModel(self.test2Model)

    def test3TableView(self, df):
        """Populate the QTableView with output data (expected as a pandas DataFrame)."""
        # Clear the existing data in the model
        self.test3Model.clear()

        # Ensure df is a DataFrame; if not, convert or handle appropriately
        if not isinstance(df, pd.DataFrame):
            logger.info("Data is not a DataFrame.")
            return

        headers = df.columns.tolist()

        # Set headers
        self.test3Model.setHorizontalHeaderLabels(headers)

        # Populate the model with data
        for index, row_data in df.iterrows():
            row = []
            for item in row_data:
                row.append(QStandardItem(str(item)))  # Ensure everything is converted to string
            self.test3Model.appendRow(row)

        # Update the table view
        self.tableView_3.setModel(self.test3Model)

    def test4TableView(self, df):
        """Populate the QTableView with output data (expected as a pandas DataFrame)."""
        # Clear the existing data in the model
        self.test4Model.clear()

        # Ensure df is a DataFrame; if not, convert or handle appropriately
        if not isinstance(df, pd.DataFrame):
            logger.info("Data is not a DataFrame.")
            return

        headers = df.columns.tolist()

        # Set headers
        self.test4Model.setHorizontalHeaderLabels(headers)

        # Populate the model with data
        for index, row_data in df.iterrows():
            row = []
            for item in row_data:
                row.append(QStandardItem(str(item)))  # Ensure everything is converted to string
            self.test4Model.appendRow(row)

        # Update the table view
        self.tableView_4.setModel(self.test4Model)