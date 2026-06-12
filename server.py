from __future__ import annotations
from typing import Optional
from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel, Field, model_validator
