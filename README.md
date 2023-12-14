
# MLOps Project

This project is aimed at establishing an MLOps pipeline for training an image captioning model for diffusion model learning.

## Pipeline Configuration

The current pipelines in use:

1. **pipeline_0000-prepare_data:**
    - A pipeline for preprocessing data, specifically tailored for the diffusion model learning image captioning.
    - Tasks include data loading, handling missing values, feature engineering, and saving processed data.

2. **pipeline_0001-train_model:**
    - A pipeline designed for training the image captioning model within the context of diffusion model learning.
    - Involves loading preprocessed data, training the image captioning model, and saving the trained model.

3. **pipeline_0003-evaluate_model:**
    - A pipeline dedicated to evaluating the trained image captioning model for diffusion model learning.
    - Tasks include loading test data, evaluating the image captioning model, and saving the results.

4. **pipeline_0004-deploy_model:**
    - A pipeline focused on deploying the trained image captioning model tailored for diffusion model learning.
    - Involves loading the trained model, deploying it, and configuring the environment.



# Lisence
```
MIT License

Copyright (c) 2023 JeongYuH

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```