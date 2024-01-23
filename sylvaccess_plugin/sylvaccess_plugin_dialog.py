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
#from skidder import Skidder
#from porteur import Porteur
#from cable import Cable
#from cable_opti import Cable_opti
#from error import Error

# Chargement de l'interface utilisateur depuis le fichier .ui
FORM_CLASS, _ = uic.loadUiType(os.path.join(os.path.dirname(__file__), 'sylvaccess_plugin_dialog_base.ui'))

class Sylvaccess_pluginDialog(QtWidgets.QDialog, FORM_CLASS):
    def __init__(self, parent=None):
        super(Sylvaccess_pluginDialog, self).__init__(parent)
        self.setupUi(self)

        # Connexion des signaux des boutons d'ouverture de fichier à la fonction open_folder
        for i in range(1, 14):
            button = getattr(self, f"pushButton_{i}")
            button.clicked.connect(lambda _, num=i: self.open_folder(num))

        # Connexion des signaux des checkbox
        for i in range(1, 5):
            checkbox = getattr(self, f"checkBox_{i}")
            checkbox.stateChanged.connect(lambda _, num=i: self.checkbox_state_changed(num))

        # Connexion des signaux des boutons OK et Annuler
        self.button_box.accepted.connect(self.launch)
        self.button_box.rejected.connect(self.reject)

    def open_folder(self, button_number):    
        # Définit les filtres génériques pour Shapefiles et fichiers raster
        shapefile_filter = "Shapefiles (*.shp);;All files (*)"
        raster_filter = "Raster files (*.tif *.asc *.txt);;All files (*)"

        # Définit les options de la boîte de dialogue
        options = QFileDialog.Options()
        options |= QFileDialog.DontUseNativeDialog

        # Affiche le dialogue de sélection de fichier avec les filtres appropriés
        if button_number in [4, 5, 6]:
            selected_file, _ = QFileDialog.getOpenFileName(
                None, "Choisir un fichier", filter=shapefile_filter, options=options)
        elif button_number in [3, 11, 12, 13]:
            selected_file, _ = QFileDialog.getOpenFileName(
                None, "Choisir un fichier", filter=raster_filter, options=options)
        elif button_number in [1, 2, 7, 8, 9, 10]:  # Pour le bouton qui doit ouvrir un dossier
            selected_file = QFileDialog.getExistingDirectory(
                None, "Choisir un dossier", options=options)

        if selected_file:
            # Mise à jour du champ de texte approprié
            if button_number == 2:
                text_edit = getattr(self, f"lineEdit_2")
                text_edit.setText(selected_file)
                text_edit = getattr(self, f"lineEdit_17")
                text_edit.setText(selected_file)
            elif button_number == 4 :
                text_edit = getattr(self, f"lineEdit_4")
                text_edit.setText(selected_file)
                text_edit = getattr(self, f"lineEdit_14")
                text_edit.setText(selected_file)
            elif button_number == 13:
                text_edit = getattr(self, f"lineEdit_13")
                text_edit.setText(selected_file)
                text_edit = getattr(self, f"lineEdit_15")
                text_edit.setText(selected_file)
            elif button_number == 12 :
                text_edit = getattr(self, f"lineEdit_12")
                text_edit.setText(selected_file)
                text_edit = getattr(self, f"lineEdit_16")
                text_edit.setText(selected_file)
            else:
                text_edit = getattr(self, f"lineEdit_{button_number}")
                text_edit.setText(selected_file)

    def checkbox_state_changed(self, checkbox_number):
        # Récupère l'état de la checkbox
        checkbox = getattr(self, f"checkBox_{checkbox_number}")
        checkbox_state = checkbox.isChecked()

        if checkbox_state:
            if checkbox_number == 1:
                self.cable_opti.setEnabled(True)
                # désactive les lineEdis pour éviter les erreurs
                self.lineEdit_14.setEnabled(False)
                self.lineEdit_15.setEnabled(False)
                self.lineEdit_16.setEnabled(False)
                self.lineEdit_17.setEnabled(False)
            if checkbox_number == 2:
                self.cable.setEnabled(True)
            if checkbox_number == 3:
                self.porteur.setEnabled(True) 
            if checkbox_number == 4:
                self.skidder.setEnabled(True)

    def launch(self):
        
        for i in range (1,6):
            if f"lineEdit_{i}".text() == '':
                #Error("Veuillez remplir tous les champs")
                return
            elif getattr(f"checkBox_4".isChecked()):
                #Skidder(f"lineEdit_{i}".text())
                return
            elif getattr(f"checkBox_3".isChecked()):
                #Porteur(f"lineEdit_{i}".text())
                return
            elif getattr(f"checkBox_2".isChecked()):
                #Cable(f"lineEdit_{i}".text())
                return
            elif getattr(f"checkBox_1".isChecked()):
                #Cable_opti(f"lineEdit_{i}".text())
                return
            else:
                #Error("Veuillez choisir au moins un type de machine")
                return


