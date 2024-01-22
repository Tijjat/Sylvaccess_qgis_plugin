# -*- coding: utf-8 -*-
"""
/***************************************************************************
 Sylvaccess_pluginDialog
                                 A QGIS plugin
 This plugin is the Sylvaccess app made in qgis
 Generated by Plugin Builder: http://g-sherman.github.io/Qgis-Plugin-Builder/
                             -------------------
        begin                : 2024-01-19
        git sha              : $Format:%H$
        copyright            : (C) 2024 by Cosylval
        email                : yoann.zenner@viacesi.fr
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

from PyQt5 import QtWidgets, uic
from PyQt5.QtWidgets import QFileDialog
import os

# Chargement de l'interface utilisateur depuis le fichier .ui
FORM_CLASS, _ = uic.loadUiType(os.path.join(os.path.dirname(__file__), 'sylvaccess_plugin_dialog_base.ui'))

class Sylvaccess_pluginDialog(QtWidgets.QDialog, FORM_CLASS):
    def __init__(self, parent=None):
        super(Sylvaccess_pluginDialog, self).__init__(parent)
        self.setupUi(self)

        # Connexion des signaux des boutons à la fonction open_folder
        for i in range(1, 18):
            button = getattr(self, f"pushButton_{i}")
            button.clicked.connect(lambda _, num=i: self.open_folder(num))

        self.button_box.accepted.connect(self.accept)
        self.button_box.rejected.connect(self.reject)

    def open_folder(self, button_number):    
        # Définit les filtres génériques pour Shapefiles et fichiers raster
        shapefile_filter = "Shapefiles (*.shp);;All files (*)"
        raster_filter = "Raster files (*.tif *.asc *.txt);;All files (*)"

        # Définit les options de la boîte de dialogue
        options = QFileDialog.Options()
        options |= QFileDialog.DontUseNativeDialog

        # Affiche le dialogue de sélection de fichier avec les filtres appropriés
        if button_number in [4, 5, 6, 14]:
            selected_file, _ = QFileDialog.getOpenFileName(
                None, "Choisir un fichier", filter=shapefile_filter, options=options)
        elif button_number in [3, 11, 12, 13, 15, 16]:
            selected_file, _ = QFileDialog.getOpenFileName(
                None, "Choisir un fichier", filter=raster_filter, options=options)
        elif button_number in [1, 2, 7, 8, 9, 10, 17]:  # Pour le bouton qui doit ouvrir un dossier
            selected_file = QFileDialog.getExistingDirectory(
                None, "Choisir un dossier", options=options)

        if selected_file:
            # Mise à jour du champ de texte approprié
            text_edit = getattr(self, f"lineEdit_{button_number}")
            text_edit.setText(selected_file)
