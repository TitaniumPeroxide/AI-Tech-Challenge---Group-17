let currentSectionIndex = 0;
let totalScore = 0;

const sectionOrder = ["Age", "Visa/Residency Status", "Duration in Ireland"];
const DOM = {
  chatbox: document.getElementById("chatbox"),
  inputField: document.getElementById("userInput"),
  stepIndicator: document.getElementById("current-step"),
  totalSteps: document.getElementById("total-steps"),
  typingIndicator: document.getElementById("typing-indicator"),
};

DOM.totalSteps.textContent = sectionOrder.length;

window.onload = () => {
  displayMessage("👋 Welcome to the Loan Evaluation Assistant!", "bot");
  loadNextQuestion();
};

DOM.inputField.addEventListener("keydown", (e) => {
  if (e.key === "Enter" && !e.shiftKey) {
    e.preventDefault();
    handleUserInput();
  }
});

async function handleUserInput() {
  const userInput = DOM.inputField.value.trim();
  if (!userInput) return;

  displayMessage(userInput, "user");
  DOM.inputField.value = "";

  const section = sectionOrder[currentSectionIndex];
  showTyping(true);

  try {
    const response = await fetch("https://humble-umbrella-466vr679j6rfw46-5000.app.github.dev/evaluate", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ section, question: section, answer: userInput }),
    });

    if (!response.ok) throw new Error("Server error");

    const { section: sec, answer, score, reason, eligability } = await response.json();
    totalScore += score;

    displayMessage(
      `✅ **${sec}**\nAnswer: ${answer}\nScore: ${score}\nReason: ${reason}\nEligibility: ${eligability}`,
      "bot"
    );

    currentSectionIndex++;
    DOM.stepIndicator.textContent = currentSectionIndex + 1;

    currentSectionIndex < sectionOrder.length
      ? loadNextQuestion()
      : showFinalResult();
  } catch (err) {
    console.error(err);
    displayMessage("⚠️ Error connecting to the server. Please try again later.", "bot");
  } finally {
    showTyping(false);
  }
}

async function loadNextQuestion() {
  const section = sectionOrder[currentSectionIndex];
  showTyping(true);

  try {
    const res = await fetch("https://humble-umbrella-466vr679j6rfw46-5000.app.github.dev/get_question", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ section }),
    });

    if (!res.ok) throw new Error("Server error");

    const { question } = await res.json();
    displayMessage(question, "bot");
  } catch (err) {
    console.error(err);
    displayMessage("⚠️ Failed to load the question.", "bot");
  } finally {
    showTyping(false);
  }
}

function showFinalResult() {
  const resultMessage = totalScore > 50 ? "🎉 Loan Approved!" : "❌ Loan Denied.";
  displayMessage(resultMessage, "bot");
  displayMessage(`Final Score: ${totalScore}`, "bot");
}

function displayMessage(msg, sender) {
  const msgDiv = document.createElement("div");
  msgDiv.className = `message ${sender}`;
  msgDiv.innerText = msg;
  DOM.chatbox.appendChild(msgDiv);
  DOM.chatbox.scrollTop = DOM.chatbox.scrollHeight;
}

function showTyping(visible) {
  DOM.typingIndicator.style.display = visible ? "block" : "none";
}
