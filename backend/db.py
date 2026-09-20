"""
RetinaSetu - Clinical Database Adapter (MongoDB with Local Document Store Fallback)
Handles:
  1. Patient Demographics & Intake
  2. Screening Records & Lesion Biomarkers
  3. Doctor Review & Sign-Off Queue
"""

import os
import json
import uuid
from datetime import datetime

DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "db")
os.makedirs(DATA_DIR, exist_ok=True)

PATIENTS_FILE = os.path.join(DATA_DIR, "patients.json")
SCREENINGS_FILE = os.path.join(DATA_DIR, "screenings.json")
REVIEWS_FILE = os.path.join(DATA_DIR, "doctor_reviews.json")

def _clean_mongo_doc(doc):
    if isinstance(doc, dict):
        cleaned = {}
        for k, v in doc.items():
            if k == "_id":
                continue
            cleaned[k] = _clean_mongo_doc(v)
        return cleaned
    elif isinstance(doc, list):
        return [_clean_mongo_doc(item) for item in doc]
    return doc

def _load_json(filepath):
    if os.path.exists(filepath):
        try:
            with open(filepath, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return []
    return []

def _save_json(filepath, data):
    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, default=str)

class DatabaseManager:
    def __init__(self, mongo_uri="mongodb://localhost:27017"):
        self.use_mongo = False
        self.mongo_client = None
        self.db = None
        try:
            import pymongo
            client = pymongo.MongoClient(mongo_uri, serverSelectionTimeoutMS=1000)
            client.server_info() # trigger connection check
            self.mongo_client = client
            self.db = client["retinasetu"]
            self.use_mongo = True
            print("[Database] Connected to live MongoDB daemon on localhost:27017")
        except Exception as e:
            self.use_mongo = False
            print(f"[Database] MongoDB server not active ({e}). Using persistent JSON document store at {DATA_DIR}")
        
        # Ensure database is clean of duplicate names on startup
        self._deduplicate_patients_collection()

    def _deduplicate_patients_collection(self):
        """
        Cleans up existing duplicates in the collection, keeping only the latest per full_name.
        """
        try:
            if self.use_mongo:
                cursor = self.db.patients.find().sort("created_at", -1)
                seen = set()
                ids_to_remove = []
                for p in cursor:
                    name_key = p.get("full_name", "").strip().lower()
                    if not name_key or name_key in seen:
                        ids_to_remove.append(p["_id"])
                    else:
                        seen.add(name_key)
                if ids_to_remove:
                    self.db.patients.delete_many({"_id": {"$in": ids_to_remove}})
            else:
                records = _load_json(PATIENTS_FILE)
                seen = set()
                unique_records = []
                for p in records:
                    name_key = p.get("full_name", "").strip().lower()
                    if name_key and name_key not in seen:
                        seen.add(name_key)
                        unique_records.append(p)
                _save_json(PATIENTS_FILE, unique_records)
        except Exception as e:
            print(f"[Database] Patient deduplication note: {e}")

    def create_or_update_patient(self, patient_dict):
        """
        Stores patient demographics and clinical history.
        Enforces unique full_name (case-insensitive): if a patient with the same name exists,
        updates the existing record rather than creating a duplicate.
        """
        raw_name = patient_dict.get("full_name", "").strip()
        patient_id = patient_dict.get("patient_id")

        patient_dict["updated_at"] = datetime.now().isoformat()
        if "created_at" not in patient_dict:
            patient_dict["created_at"] = datetime.now().isoformat()

        if self.use_mongo:
            existing = None
            if raw_name:
                import re
                existing = self.db.patients.find_one({"full_name": {"$regex": f"^{re.escape(raw_name)}$", "$options": "i"}})
            elif patient_id:
                existing = self.db.patients.find_one({"patient_id": patient_id})

            if existing:
                target_id = existing.get("patient_id") or patient_id
                patient_dict["patient_id"] = target_id
                p_to_save = dict(patient_dict)
                if "_id" in p_to_save:
                    del p_to_save["_id"]
                self.db.patients.update_one({"_id": existing["_id"]}, {"$set": p_to_save})
                return _clean_mongo_doc(patient_dict)
            else:
                if not patient_id:
                    patient_id = f"RS-PHC-{datetime.now().strftime('%Y%m')}-{uuid.uuid4().hex[:4].upper()}"
                    patient_dict["patient_id"] = patient_id
                p_to_save = dict(patient_dict)
                if "_id" in p_to_save:
                    del p_to_save["_id"]
                self.db.patients.update_one({"patient_id": patient_id}, {"$set": p_to_save}, upsert=True)
                return _clean_mongo_doc(patient_dict)
        else:
            records = _load_json(PATIENTS_FILE)
            idx = None
            if raw_name:
                idx = next((i for i, r in enumerate(records) if r.get("full_name", "").strip().lower() == raw_name.lower()), None)
            if idx is None and patient_id:
                idx = next((i for i, r in enumerate(records) if r.get("patient_id") == patient_id), None)

            if idx is not None:
                patient_dict["patient_id"] = records[idx].get("patient_id", patient_id)
                records[idx] = patient_dict
            else:
                if not patient_id:
                    patient_id = f"RS-PHC-{datetime.now().strftime('%Y%m')}-{uuid.uuid4().hex[:4].upper()}"
                    patient_dict["patient_id"] = patient_id
                records.insert(0, patient_dict)
            _save_json(PATIENTS_FILE, records)
            return _clean_mongo_doc(patient_dict)

    def get_patient(self, patient_id):
        if self.use_mongo:
            doc = self.db.patients.find_one({"patient_id": patient_id})
            return _clean_mongo_doc(doc)
        else:
            records = _load_json(PATIENTS_FILE)
            doc = next((r for r in records if r.get("patient_id") == patient_id), None)
            return _clean_mongo_doc(doc)

    def list_patients(self, limit=50):
        """
        Lists patients ensuring each unique full_name is only returned once (most recent encounter).
        """
        if self.use_mongo:
            cursor = self.db.patients.find().sort("created_at", -1)
            raw_list = _clean_mongo_doc(list(cursor))
        else:
            raw_list = _clean_mongo_doc(_load_json(PATIENTS_FILE))

        unique_patients = []
        seen_names = set()
        for p in raw_list:
            name_key = p.get("full_name", "").strip().lower()
            if not name_key:
                continue
            if name_key not in seen_names:
                seen_names.add(name_key)
                unique_patients.append(p)
                if len(unique_patients) >= limit:
                    break

        return unique_patients

    def save_screening(self, screening_dict):
        """
        Saves full screening inference result tied to a patient.
        """
        screening_id = screening_dict.get("screening_id")
        if not screening_id:
            screening_id = f"SCR-{datetime.now().strftime('%Y%m%d')}-{uuid.uuid4().hex[:6].upper()}"
            screening_dict["screening_id"] = screening_id

        screening_dict["timestamp"] = datetime.now().isoformat()
        if "review_status" not in screening_dict:
            # If referable, needs doctor review
            is_referable = screening_dict.get("grading", {}).get("is_referable", False)
            screening_dict["review_status"] = "PENDING_DOCTOR_REVIEW" if is_referable else "AUTOCLEARED"

        if self.use_mongo:
            s_to_save = dict(screening_dict)
            if "_id" in s_to_save:
                del s_to_save["_id"]
            self.db.screenings.update_one({"screening_id": screening_id}, {"$set": s_to_save}, upsert=True)
            return _clean_mongo_doc(screening_dict)
        else:
            records = _load_json(SCREENINGS_FILE)
            idx = next((i for i, r in enumerate(records) if r.get("screening_id") == screening_id), None)
            if idx is not None:
                records[idx] = screening_dict
            else:
                records.insert(0, screening_dict)
            _save_json(SCREENINGS_FILE, records)
            return _clean_mongo_doc(screening_dict)

    def get_screening(self, screening_id):
        if self.use_mongo:
            doc = self.db.screenings.find_one({"screening_id": screening_id})
            return _clean_mongo_doc(doc)
        else:
            records = _load_json(SCREENINGS_FILE)
            doc = next((r for r in records if r.get("screening_id") == screening_id), None)
            return _clean_mongo_doc(doc)

    def list_screenings(self, status=None, limit=50):
        if self.use_mongo:
            query = {"review_status": status} if status else {}
            cursor = self.db.screenings.find(query).sort("timestamp", -1).limit(limit)
            return _clean_mongo_doc(list(cursor))
        else:
            records = _load_json(SCREENINGS_FILE)
            if status:
                records = [r for r in records if r.get("review_status") == status]
            return _clean_mongo_doc(records[:limit])

    def submit_review(self, screening_id, doctor_notes, doctor_name, confirmed_grade, action):
        """
        Records human ophthalmologist verification & sign-off.
        """
        review_doc = {
            "review_id": f"REV-{uuid.uuid4().hex[:6].upper()}",
            "screening_id": screening_id,
            "doctor_name": doctor_name,
            "confirmed_grade": confirmed_grade,
            "clinical_notes": doctor_notes,
            "referral_action": action,
            "signed_at": datetime.now().isoformat()
        }

        # Update screening status
        if self.use_mongo:
            r_to_insert = dict(review_doc)
            if "_id" in r_to_insert:
                del r_to_insert["_id"]
            self.db.reviews.insert_one(r_to_insert)
            # Remove any added _id
            if "_id" in r_to_insert:
                del r_to_insert["_id"]
            self.db.screenings.update_one(
                {"screening_id": screening_id},
                {"$set": {
                    "review_status": "REVIEWED",
                    "doctor_review": r_to_insert
                }}
            )
            return _clean_mongo_doc(review_doc)
        else:
            reviews = _load_json(REVIEWS_FILE)
            reviews.insert(0, review_doc)
            _save_json(REVIEWS_FILE, reviews)

            screenings = _load_json(SCREENINGS_FILE)
            for s in screenings:
                if s.get("screening_id") == screening_id:
                    s["review_status"] = "REVIEWED"
                    s["doctor_review"] = review_doc
                    break
            _save_json(SCREENINGS_FILE, screenings)
            return _clean_mongo_doc(review_doc)

db_manager = DatabaseManager()
