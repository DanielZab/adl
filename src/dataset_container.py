from typing import List, Union
import logging
import datetime

class DataPoint:
    #{"v": 7286, "vw": 13.2226, "o": 13.23, "c": 13.2215, "h": 13.23, "l": 13.21, "t": 1728658800000, "n": 29}
    def __init__(self, data: dict):
        assert set(data.keys()) == {"v", "vw", "o", "c", "h", "l", "t", "n"}
        for k, v in data.items():
            setattr(self, k, v)

    def __lt__(self, other):
        if isinstance(other, datetime.datetime):
            return self.t < other.timestamp() * 1000

        return self.t < other.t
    
    def __gt__(self, other):
        if isinstance(other, datetime.datetime):
            return self.t > other.timestamp() * 1000

        return self.t > other.t
    
    def __str__(self):
        msg = "DataPoint {"
        for k, v in self.__dict__.items():
            msg += f"{k}: {v}, "
        return msg[:-2] + "}"
    
    def __repr__(self):
        return str(self)
    
    def price(self):
        return self.o

class DataSet:
    def __init__(self, name: str, data_points: Union[List[DataPoint], List[dict]] = []):
        self.name = name

        if len(data_points) > 0 and all(isinstance(e, DataPoint) for e in data_points):
            self.data_points = data_points
        else:
            self.data_points = list(map(lambda x: DataPoint(x), data_points))

    def insert(self, data_point: Union[DataPoint, List[DataPoint], dict, List[dict]]):

        if isinstance(data_point, list):
            for dp in data_point:
                self.insert(dp)
            return

        if isinstance(data_point, dict):
            data_point = DataPoint(data_point)

        assert isinstance(data_point, DataPoint)

        if len(self.data_points) == 0:
            self.data_points.append(data_point)
            return

        if self.data_points[-1] < data_point:
            self.data_points.append(data_point)
        else:
            logging.DEBUG(f"DataPoint {data_point} not inserted at last position")
            for i in range(len(self.data_points)):
                if self.data_points[i] > data_point:
                    self.data_points.insert(i, data_point)
                    break
    
    def __str__(self):
        msg = "DataSet {" + ",\n".join(map(str, self.data_points)) + "}"
        return msg
    
    def __repr__(self):
        return str(self)

    def __getitem__(self, key):
        return self.data_points[key]
    
    def __len__(self):
        return len(self.data_points)
    
    def train_validation_test_split(self, split_date1: datetime.datetime, split_date2: datetime.datetime):
        assert split_date1 < split_date2
        assert self.data_points[0] < split_date1
        assert self.data_points[-1] > split_date2
    
        index1 = -1
        index2 = -1
        for i, e in enumerate(self.data_points):
            if index1 == -1 and e > split_date1:
                index1 = i
            if e > split_date2:
                index2 = i
                break
        assert index1 != -1 and index2 != -1
        return DataSet(self.name, self.data_points[:index1]), DataSet(self.name, self.data_points[index1:index2]), DataSet(self.name, self.data_points[index2:])
                
    


