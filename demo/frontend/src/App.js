import React, { Component } from "react";
import axios from "axios";

class App extends Component {
  constructor(props) {
    super(props);
    this.state = {
      articleURL: "",
      imagePosition: 1,
      isLoaded: false,
      isLoading: false,
      title: "",
      imageURL: "",
      start: "",
      before: "",
      after: "",
      trueCaption: "",
      generatedCaption: "",
      hasError: false,
      errorMessage: ""
    };
  }

  componentDidMount() {
    document.body.classList.add("bg-light");
  }

  fetchCaption = e => {
    e.preventDefault();
    this.setState({ isLoaded: false, isLoading: true, hasError: false });
    const query = {
      url: this.state.articleURL,
      pos: this.state.imagePosition
    };
    axios
      .post("/api/caption/", query)
      .then(res => {
        if (res.data.error) {
          this.setState({
            isLoading: false,
            hasError: true,
            isLoaded: false,
            errorMessage: res.data.error
          });
        } else {
          this.setState({
            isLoaded: true,
            isLoading: false,
            title: res.data.title,
            imageURL: res.data.image_url,
            start: res.data.start,
            before: res.data.before,
            after: res.data.after,
            trueCaption: res.data.true_caption,
            generatedCaption: res.data.generated_caption
          });
        }
      })
      .catch(function(error) {
        console.log(error);
      });
  };

  handleURLChange = e => {
    this.setState({ articleURL: e.target.value });
  };

  selectArticle = e => {
    this.setState({ articleURL: e.target.getAttribute("url") });
  };

  handleImagePositionChange = e => {
    console.log(e.target.value);
    this.setState({ imagePosition: e.target.value });
  };

  splitNewLines = text =>
    text.split("\n").map((item, key, arr) => (
      <span key={key}>
        {item}
        {arr.length - 1 === key ? (
          <div />
        ) : (
          <div>
            <br />
            <br />
          </div>
        )}
      </span>
    ));

  render() {
    return (
      <div className="container">
        <div className="py-5">
          <h2 className="text-center">Transform and Tell</h2>
          <p className="lead text-center">
            Demo accompanying the paper{" "}
            <em>Transform and Tell: Entity-Aware News Image Captioning</em>.
          </p>
          <p>
            Click on one of the following examples that was used in the paper:
          </p>
          <div className="list-group">
            <button
              type="button"
              className="list-group-item list-group-item-action"
              onClick={this.selectArticle}
              url="https://www.nytimes.com/2019/08/07/t-magazine/theresa-chromati-artist.html"
            >
              An Artist Making a Powerful Statement — by Creating Work About
              Herself
            </button>
            <button
              type="button"
              className="list-group-item list-group-item-action"
              onClick={this.selectArticle}
              url="https://www.nytimes.com/2019/06/11/sports/womens-world-cup-usa-soccer.html"
            >
              What a 13-0 U.S. Win Over Thailand Looked Like at the Women’s
              World Cup
            </button>
          </div>
          <br />
          <form>
            <div className="form-group">
              <label htmlFor="articleURL">
                Or manually provide the URL to a New York Times article:
              </label>
              <input
                type="url"
                className="form-control"
                id="articleURL"
                aria-describedby="urlHelp"
                placeholder="Article URL"
                value={this.state.articleURL}
                onChange={this.handleURLChange}
              />
            </div>
            <div className="form-group">
              <label for="imagePosition">
                Choose an image position (e.g. choose 1 if we want to write a
                caption for the top image):
              </label>
              <select
                className="form-control"
                id="imagePosition"
                onChange={this.handleImagePositionChange}
              >
                <option>1</option>
                <option>2</option>
                <option>3</option>
                <option>4</option>
                <option>5</option>
              </select>
            </div>
            <button
              type="submit"
              className="btn btn-lg btn-primary"
              onClick={this.fetchCaption}
              disabled={this.state.isLoading}
            >
              {this.state.isLoading ? "Running model..." : "Generate Caption"}
            </button>
          </form>
        </div>
        {this.state.hasError && (
          <div class="alert alert-danger" role="alert">
            {this.state.errorMessage}
          </div>
        )}

        {this.state.isLoaded && (
          <div className="row">
            <div className="col-md-6 mb-4 alert alert-secondary">
              <h4 className="mb-3">{this.state.title}</h4>

              <div className="mb-3">
                {this.splitNewLines(this.state.start)}
                {this.splitNewLines(this.state.before)}
              </div>
              <div className="mb-3">
                <img src={this.state.imageURL} className="img-fluid" alt="" />
              </div>
              <div className="mb-3">{this.splitNewLines(this.state.after)}</div>
            </div>

            <div className="col-md-6 mb-4">
              {/* <h4 className="mb-3">Ground-truth caption</h4>
              <div className="mb-3">{this.state.trueCaption}</div> */}
              <div className="alert alert-success">
                <h4 className="mb-3">Generated caption</h4>
                <div className="mb-3">{this.state.generatedCaption}</div>
              </div>
            </div>
          </div>
        )}
      </div>
    );
  }
}
export default App;
