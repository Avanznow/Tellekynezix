import os
import time
from PySide6.QtCore import QObject, Slot, Property, Signal
from Developers import devCharts  # Ensure this import stays!

class DevelopersAPI(QObject):
    # Signal to tell QML that the images have been updated
    pathsChanged = Signal()
    devListChanged = Signal()

    def __init__(self):
        super().__init__()
        self._gold_path = ""
        self._silver_path = ""
        self._bronze_path = ""
        self._medal_path = ""
        self._dev_list = "Loading developers list..."
        self._loaded_once = False
        self.devImagePath()

    @Slot(result=str)
    def getDevList(self):
        return self._dev_list
        # return devCharts.devList(owner="Avanznow", repo="Tellekynezix")

    @Slot(result=str)
    def getTicketsByDev(self) -> str:
        return devCharts.ticketsByDev_text()

    @Slot()
    def devChart(self):
        if self._loaded_once:
            return
        self._loaded_once = True
        self._runChartsAndList()
    
    @Slot()
    def devChartForce(self):
        self._runChartsAndList()

    def _runChartsAndList(self):
        print("Generating charts and developers list...")
        self._dev_list = "Loading developers list..."
        self.devListChanged.emit()
        try:
            devCharts.main()

            self.devImagePath()
            self.pathsChanged.emit()

            self._dev_list = devCharts.devList(owner="Avanznow", repo="Tellekynezix")
            self.devListChanged.emit()

            print("Charts and developers list updated.")
        except Exception as e:
            print(f"Error generating charts/dev list: {e}")
            self._dev_list = "Error loading developers."
            self.devListChanged.emit()

    def devImagePath(self):
        # Anchor to the directory where this API file sits
        base_dir = os.path.dirname(os.path.abspath(__file__))
        # Point to the Developers/plotDevelopers subfolder
        plots_dir = os.path.join(base_dir, "Developers", "plotDevelopers")
        
        # Unique timestamp forces QML to bypass its image cache
        timestamp = int(time.time())
        
        def format_path(filename):
            full_p = os.path.abspath(os.path.join(plots_dir, filename)).replace("\\", "/")
            # Use file:/// for QML local file access
            return f"file:///{full_p}?t={timestamp}"

        self._gold_path = format_path("gold_contributors.png")
        self._silver_path = format_path("silver_contributors.png")
        self._bronze_path = format_path("bronze_contributors.png")
        self._medal_path = format_path("Medal.png")

    # Properties with 'notify' decorators so QML updates automatically
    @Property(str, notify=pathsChanged)
    def goldPath(self):
        return self._gold_path

    @Property(str, notify=pathsChanged)
    def silverPath(self):
        return self._silver_path

    @Property(str, notify=pathsChanged)
    def bronzePath(self):
        return self._bronze_path

    @Property(str, notify=pathsChanged)
    def medalPath(self):
        return self._medal_path

    @Property(str, notify=devListChanged)
    def devListText(self):
        return self._dev_list
