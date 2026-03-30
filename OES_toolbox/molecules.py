import os
import datetime
from pathlib import Path
from pathlib import Path
import numpy as np
from scipy.signal import fftconvolve

from PyQt6.QtWidgets import QFileDialog, QTreeWidgetItemIterator, QTableWidgetItem, \
        QMessageBox, QCheckBox, QMenu
from PyQt6.QtCore import Qt, QObject, QThread, pyqtSignal
from PyQt6.QtGui import QAction
from PyQt6 import QtGui

from .Widgets import MoleculeCheckBox
from .lazy_import import lazy_import
scipy = lazy_import("scipy")
Moose = lazy_import("Moose")
# Exclude high J Swan band database; can quickly run OOM without care
MOLECULES = [f for f in Moose.database_files if "J300" not in f]
MOLECULE_DB_LABELS = {
    "C2_swan":"C₂ Swan",
    "CNBX": "CN (B-X)",
    "N2CB": "N₂ (C-B)",
    "N2PlusBX": "N₂⁺ (B-X)",
    "NHAX": "NH (A-X)",
    "NOBX": "NO (B-X)",
    "OHAX": "OH (A-X)"
}
"""Mapping of database names to better looking label for use in UI."""

LIFBASE_SIMS = [f.stem for f in Path(f"{__file__}").parent.joinpath("data/mol_spec").glob("*.mod")]
LIFBASE_LABELS = {
    'NOAX': "NO (A-X)",
    'NOCX': "NO (C-X)",
    'NODX': "NO (D-X)",
    'CHAX': "CH (A-X)",
    'CHBX': "CH (B-X)",
    'CHCX': "CH (C-X)",
    'SiHAX': "SiH (A-X)",
    'CFAX': "CF (A-X)",
    'CFBX': "CF (B-X)",
    'CFCX': "CF (C-X)",
    'CFDX': "CF (D-X)",
    'CNBX': "CN (B-X)"
}


def model_for_fit(x, T_rot, T_vib, sim_db, instr, resolution=1000, wl_pad=10):
    """Function copied from Moose without the normalization to the maximum. """
    sticks = Moose.create_stick_spectrum(T_vib, T_rot, df_db=sim_db)
    refined = Moose.equidistant_mesh(sticks, wl_pad=wl_pad, resolution=resolution)
    simulation = apply_voigt(refined, instr)
    sim_matched = match_spectra(x.reshape(-1, 1), simulation)
    return sim_matched[:, 1]

def apply_voigt(sim, instr):
    """Function copied from Moose to allow arbitrary instrumental functions."""
    from scipy.signal import fftconvolve
    x = sim[:, 0]
    conv = scipy.signal.fftconvolve(sim[:, 1], instr(x), mode="same")
    return np.array([x, conv]).T

def match_spectra(meas, sim):
    """Function copied from Moose to solve out of bounds errors."""
    interp = scipy.interpolate.make_interp_spline(sim[:, 0], sim[:, 1])
    interp.extrapolate = False
    matched_y = interp(meas[:, 0])
    matched_y = np.nan_to_num(matched_y)
    return np.array([meas[:, 0], matched_y]).T

def get_mOES_spec(x, Tvib, Trot, data, instr):
    # TODO: explictily handle errors now that all exceptions are not silently ignored.
    sim_y = model_for_fit(x, Trot, Tvib, data, instr)
    return sim_y/np.sum(sim_y)


class MoleculeFitter(QObject):
    finished = pyqtSignal()
    result_ready = pyqtSignal(str, lmfit.minimizer.MinimizerResult, np.ndarray, np.ndarray)
    progress = pyqtSignal(int)

    
    def __init__(self, label, x, y, T_rot,T_vib, molecule_dbs:dict[str,"DataFrame"], sep_Trot:bool, sep_Tvib:bool, instr_func, 
                                            allow_shift=False, allow_stretch=False, parent=None):
        super(self.__class__, self).__init__(parent)
        self.label = label
        self.x, self.y = x,y
        self.T_rot = T_rot # initial value
        self.T_vib = T_vib # initial value
        self.molecule_dbs = molecule_dbs
        self.sep_Trot = sep_Trot
        self.sep_Tvib = sep_Tvib
        self.get_instr = instr_func # function copy from Window class
        self.stop = False # stop flag invoked by button press
        self.shift, self.stretch = shift, stretch

    finished = pyqtSignal()
    result_ready = pyqtSignal(str, np.ndarray, np.ndarray, np.ndarray)
    # data_ready = pyqtSignal(str, astropy.table.table.Table)
    progress = pyqtSignal(int)


    def fitfunc(self, x, *args):
        p0 = list(args) # needed for pop
        y0 = p0.pop(0)
        stretch, shift = 0, 0
        if self.stretch:
            stretch = p0.pop(-1)
        if self.shift:
            shift = p0.pop(-1)

        if not self.sep_Trot:
            Trot = p0.pop(0)
        if not self.sep_Tvib:
            Tvib = p0.pop(0)

        specs = []
        for mol_sel in self.molecules:
            if mol_sel.isChecked() and mol_sel.can_fit == True:
                A = p0.pop(0)        
                if self.sep_Trot:
                    Trot = p0.pop(0)
                if self.sep_Tvib:
                    Tvib = p0.pop(0)
                x_new = np.mean(x) + ((x - np.mean(x)) * (1 + stretch)) + shift
                db = mol_sel.get_db()
                this_spec = A*get_mOES_spec(x_new, Tvib, Trot, db, self.get_instr)
                specs.append(this_spec)
        
        return np.sum(specs, axis=0) + y0


    def fit(self):
        self.progress.emit(1)
        params=make_fit_params(
            self.y,
            species=self.molecule_dbs,
            Trot = self.T_rot,
            Tvib=self.T_vib,
            mu = 0, # Plot data is shifted already
            vary_shift = self.allow_shift,
            separate_Tvib=self.sep_Tvib, 
            separate_Trot=self.sep_Trot,
            vary_broadening=True
        )
        # TODO: investigate if error handling is needed.
        result = lmfit.minimize(
            multi_species_objective,
            params,
            args=(self.x,),
            kws=
            {
                "y":self.y,
                "normalize":True,
                **self.molecule_dbs
            },
            ftol=1e-10,
            max_nfev = 2000
        )
        y_fit = multi_species_objective(result.params, x = self.x, normalize = True, **self.molecule_dbs)
        self.result_ready.emit(self.label, result, self.x, y_fit)
        self.progress.emit(-1)
        self.finished.emit()



##############################################################################
# <------------------------- molecules module -----------------------------> #
##############################################################################   
class molecule_module:
    def __init__(self, mainWindow):
        self.mw = mainWindow
        self.get_instr = self.mw.settings.get_instr 
        molecule_list_fit = [{"ident":k,"label":MOLECULE_DB_LABELS.get(k,k), "src":"mOES"} for k in MOLECULES]
        molecule_list_no_fit = [{'ident':k, "label":LIFBASE_LABELS.get(k,k), "src":"LIFBASE"} for k in LIFBASE_SIMS]
        
        self.molecule_selectors:list[MoleculeCheckBox] = []
      
        for i,molecule in enumerate(molecule_list_fit):
            row,col = divmod(i,3)
            this_mol_check = MoleculeCheckBox(**molecule, parent=self.mw)
            this_mol_check.stateChanged.connect(self.change_sel)
            self.molecule_selectors.append(this_mol_check)
            self.mw.mol_select_grid.addWidget(this_mol_check, row, col)

        for i,molecule in enumerate(molecule_list_no_fit):
            row, col = divmod(i,3)
            this_mol_check = MoleculeCheckBox(**molecule, parent=self.mw)
            this_mol_check.stateChanged.connect(self.change_sel)
            self.molecule_selectors.append(this_mol_check)
            self.mw.mol_select_grid_nofit.addWidget(this_mol_check, row, col)
   
        self.mol_fit_threads = []
        self.mol_fit_workers = []
        

    def fitfunc(self, x, *args):
        p0 = list(args) # needed for pop
        y0 = p0.pop(0)

        if not self.mw.mol_multifit_rot_check.isChecked():
            Trot = p0.pop(0)
        if not self.mw.mol_multifit_vib_check.isChecked():
            Tvib = p0.pop(0)

        specs = []
        for mol_sel in self.molecule_selectors:
            if mol_sel.isChecked() and mol_sel.can_fit is True:
                A = p0.pop(0)        
                if self.mw.mol_multifit_rot_check.isChecked():
                    Trot = p0.pop(0)
                if self.mw.mol_multifit_vib_check.isChecked():
                    Tvib = p0.pop(0)
                this_spec = A*get_mOES_spec(x, Tvib, Trot, mol_sel.get_db(), self.get_instr)
                specs.append(this_spec)
        
        return np.sum(specs, axis=0) + y0


    def show_spec(self):
        self.clear_spec()
            
        min_x = 0
        max_x = 1100
        max_y = 1
        lim_unset = True
        Te = -1
        lw = -1
        
        
        min_x, max_x, min_y ,max_y = self.mw.get_bounds()
        
        # x = np.linspace(min_x)
        for mol_sel in self.molecule_selectors:
            if mol_sel.isChecked(): 
                db = mol_sel.get_db((min_x, max_x)) # will cache if not loaded yet
                if mol_sel.src == "mOES" and mol_sel.can_fit:
                    Trot = self.mw.mol_Trot_sbox.value()
                    Tvib = self.mw.mol_Tvib_sbox.value()
                    sim_x = np.linspace(min_x, max_x, int((max_x - min_x) * 200))
                    sim_y = get_mOES_spec(sim_x, Tvib, Trot, db, self.get_instr)
                    sim_y = sim_y / np.max(sim_y) * max_y

                    self.mw.plot(sim_x, sim_y, 'molecule: ' + mol_sel.label 
                                            + ' Tvib = ' + str(round(Tvib)) 
                                            + ' Trot = ' + str(round(Trot)) )
                        
                if mol_sel.src == "LIFBASE":
                    instr = self.get_instr(db.wl)
                    simy = scipy.signal.fftconvolve(db.I, instr / np.sum(instr), mode='same')
                    simy = simy/np.max(simy) * max_y

                    self.mw.plot(db.wl, simy, 'molecule: ' + mol_sel.label 
                                            + ' fixed temperature Tvib = 2500 K' 
                                            + ' Trot = 500 K' )

        self.mw.update_spec_colors()


    def fit_children(self,item):
        """ Recursivly walks through all children of selected tree item. Calls
        fit_filetree_item for each leaf. """
        if item.childCount() == 0:
            self.fit_filetree_item(item)
        else:
            for idx in range(item.childCount()):
                child = item.child(idx)
                self.fit_children(child) 
                
                
    def fit_filetree_item(self, this_item):
        """ Walks up the tree to assemble the path. """
        self.mw.logger.info(f"Fitting: {this_item.name()}")
        x,y = this_item.spectrum
        self.fit_spec(x,y,this_item.name())
        
    
    def fit_spec(self,x,y,label):    
        Trot0 = self.mw.mol_Trot_sbox.value()
        Tvib0 = self.mw.mol_Tvib_sbox.value()
        separate_Trot = self.mw.mol_multifit_rot_check.isChecked()
        separate_Tvib = self.mw.mol_multifit_vib_check.isChecked()

        dbs = {}
        for mol_sel in self.molecule_selectors:
            if mol_sel.isChecked() and mol_sel.can_fit is True:
                db = mol_sel.get_db()
                if db.shape[0]>0:
                    dbs[mol_sel.ident] = db

        if self.mw.mol_limit_range_check.isChecked():
            mask = (x>self.mw.mol_min_wl_sbox.value()) & (x<self.mw.mol_max_wl_sbox.value())
            y = y[mask]
            x = x[mask]

        mol_fit_thread = QThread()
        fit_worker = MoleculeFitter(
            label = label, 
            x = x, 
            y = y, 
            T_rot = Trot0,
            T_vib = Tvib0, 
            molecule_dbs = dbs,
            sep_Trot = separate_Trot, 
            sep_Tvib = separate_Tvib,
            instr_func = self.get_instr,
            allow_shift = self.mw.mol_wl_shift_check.isChecked(),
            allow_stretch = self.mw.mol_wl_stretch_check.isChecked()
        )
        
        fit_worker.moveToThread(mol_fit_thread)
        mol_fit_thread.started.connect(fit_worker.fit)
        fit_worker.result_ready.connect(self.fit_ready)
        fit_worker.progress.connect(self.mw.update_progress_bar)
        fit_worker.finished.connect(self.mw.update_spec_colors)
        fit_worker.finished.connect(fit_worker.deleteLater)
        mol_fit_thread.finished.connect(mol_fit_thread.deleteLater)
        mol_fit_thread.start()
        # We neet to store the local objects in a "self" list to ensure
        # they are not garbage collected right after the button press
        self.mol_fit_threads.append(mol_fit_thread) 
        self.mol_fit_workers.append(fit_worker)


    def fit_ready(self, label, ans:lmfit.minimizer.MinimizerResult, x_fit, y_fit):
        def get_species_name(param_name,split_count=2):
            """Construct a name for a species from a parameter name, if applicable."""
            parts = param_name.split("_", split_count)
            return MOLECULE_DB_LABELS.get(parts[split_count],parts[split_count].replace("_"," ")) if len(parts)>split_count else ""

        count = self.mw.mol_fit_results_table.rowCount()
        self.mw.mol_fit_results_table.insertRow(count)
        table = self.mw.mol_fit_results_table
        current_header = [table.horizontalHeaderItem(i).text() for i in range(table.columnCount())]

        self.mw.mol_fit_results_table.setItem(count, 0, QTableWidgetItem(label))
        header = ["file",]
        plot_label = ""
        #TODO: figure out using clean labels for the table, while mapping them to parameter names to populate cells
        print("========================")
        print("FIT result:")

        for p in ans.params:
            print(f"{p}: {ans.params[p]}")
            match p:
                case s if "fraction" in s:
                    # s = s.replace("_", " ")
                    species_name = get_species_name(s,1)
                    # col_name = f"fraction {species_name}"
                    # if col_name not in current_header:
                    #     header.append(col_name)
                    if s not in current_header:
                        header.append(s)
                case s if "T_rot" in s:
                    # s = s.replace("rot_","rot ")+" / K"
                    species_name = get_species_name(s,2)
                    # col_name = f"Trot {species_name} / K"
                    # if col_name not in current_header:
                    #     header.append(col_name)
                    if s not in current_header:
                        header.append(s)                   
                    plot_label += f"Trot {species_name}={ans.params[p].value:.0f} K "
                case s if "T_vib" in s:
                    # s = s.replace("vib_","vib ")+" / K"
                    species_name = get_species_name(s,2)
                    # col_name = f"Tvib {species_name} / K"
                    # if col_name not in current_header:
                    #     header.append(col_name) 
                    if s not in current_header:
                        header.append(s)                   
                    plot_label += f"Tvib {species_name}={ans.params[p].value:.0f} K "
                case _:
                    continue
        print("========================")
        print(header,current_header)
        if header!=current_header:
            complete_header = current_header+[h for h in header if h not in current_header]
            table.setColumnCount(len(complete_header))
            table.setHorizontalHeaderLabels(complete_header)
        else:
            complete_header = header
        
        row_idx = table.rowCount()-1
        for p in list(set(ans.params)&set(complete_header)):
            col_idx = complete_header.index(p)
            item = QTableWidgetItem()
            item.setData(Qt.ItemDataRole.DisplayRole,ans.params[p].value)
            table.setItem(row_idx,col_idx,item)

        self.mw.mol_fit_results_table.item(count, 0).y_fit = y_fit
        self.mw.mol_fit_results_table.item(count, 0).x_fit = x_fit

        # Create the plot item and associate it with the 'file name' cell as UserData, same as fit result, using `PlotItemRole`.
        # Untill we use a proper MVC pattern, this may be a good start point to show e.g. residual etc.
        plot_item = pg.PlotDataItem(x=x_fit,y=y_fit, name = f"molecule: {plot_label.strip()}")
        table.item(row_idx, 0).setData(PlotItemRole,plot_item)
        table.item(row_idx, 0).plot_item = plot_item
        table.item(row_idx, 0).setData(FitResultRole,ans)
        # TODO: consider how to avoid this loop, perhaps we need a reference to the object to be passed along?
        is_shown = f"file: {label.strip()}" in {item.name() for item in self.mw.specplot.listDataItems()}
        if is_shown:
            self.mw.specplot.addItem(plot_item, ignoreBounds=True)

                    
        self.mw.update_spec_colors()


    def fit(self):
        for plot_item in self.mw.specplot.listDataItems():
            if "molecule:" in plot_item.name():
                self.mw.specplot.removeItem(plot_item)
                
        # self.mol_fit_results_table.setRowCount(0)

        if self.mw.mol_fit_what_combobox.currentIndex() == 0: # fit all shown
            for plot_item in self.mw.specplot.listDataItems():
                if "file" in plot_item.name():
                    x,y = plot_item.getData()
                    # Remove any element that has either x or y nan
                    mask = ~np.isnan(x)& ~np.isnan(y)
                    x = x[mask]
                    y = y[mask]
                    self.fit_spec(x,y, plot_item.name().replace('file:',''))
                    
        if self.mw.mol_fit_what_combobox.currentIndex() == 1: # fit all checked
            iterator = QTreeWidgetItemIterator(self.mw.file_list,QTreeWidgetItemIterator.IteratorFlag.Checked)
            while iterator.value():
                this_item = iterator.value()
                iterator += 1
                # Fit only when not already fitted as child of parent
                if this_item.parent() is not None:
                    if not this_item.parent()._is_checked_with_ancestors():
                        self.fit_children(this_item)
                else:
                    self.fit_children(this_item) 

    def clear_spec(self):
        for plot_item in self.mw.specplot.listDataItems():
            if plot_item.name().startswith("molecule:"):
                self.mw.specplot.removeItem(plot_item)
        self.mw.update_spec_colors()
                
    def change_sel(self):
        num_checked = 0
        for mol_sel in self.molecule_selectors:
            if mol_sel.isChecked() and mol_sel.can_fit: 
            if mol_sel.isChecked() and mol_sel.can_fit: 
                num_checked = num_checked + 1
        self.mw.mol_multitemp_group.setVisible(num_checked>=2)
          
        self.mw.mol_multitemp_group.setVisible(num_checked>=2)
          
    def clear_table(self):
        self.mw.mol_fit_results_table.setRowCount(0)

    def del_table_row(self, row):
        """Delete a row from the molecure fit results table and remove the associated plot item."""
        plot_item = self.mw.mol_fit_results_table.item(row, 0).data(PlotItemRole)
        self.mw.specplot.removeItem(plot_item)
        plot_item.deleteLater()
        self.mw.mol_fit_results_table.removeRow(row)

    def fit_results_rightClick(self, cursor):
        """Show a right-click menu to interact with the fit result table."""
        row = self.mw.mol_fit_results_table.rowAt(cursor.y())
        col = self.mw.mol_fit_results_table.columnAt(cursor.x())
        item = self.mw.mol_fit_results_table.itemAt(cursor)
        # first column (i.e. the plot label) contains the extra UserRoles (FitResultRole, PlotItemRole)
        item_col0 = self.mw.mol_fit_results_table.item(row, 0)
        if item is None:
            plotted_atm = False
        else:
            plot_item = item_col0.data(PlotItemRole)
            plotted_atm = plot_item in set(self.mw.specplot.listDataItems())

        menu = QMenu()
        plot_action = QAction("Plot row", checkable=True)
        del_row_action = QAction("Remove row")
        del_col_action = QAction("Remove column")
        clear_action = QAction("Clear table")

        if plotted_atm:
            plot_action.setChecked(True)
        else:
            plot_action.setChecked(False)
        if item:
            menu.addAction(plot_action)
            plot_action.triggered.connect(lambda: self.plotl_table_item(row, not plotted_atm))
            menu.addAction(del_row_action)
            del_row_action.triggered.connect(lambda: self.del_table_row(row))
        if col !=0:
            menu.addAction(del_col_action)
        menu.addAction(clear_action)

        del_col_action.triggered.connect(lambda: self.del_table_col(col))

        clear_action.triggered.connect(self.clear_table)

        menu.exec(QtGui.QCursor.pos())


    def plotl_table_item(self, row_idx, plot:bool):
        if plot:
            self.mw.specplot.addItem(plot_item) 
        else:
            self.mw.specplot.removeItem(plot_item)
        self.mw.update_spec_colors()


    
